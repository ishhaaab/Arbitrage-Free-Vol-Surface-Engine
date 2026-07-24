"""SSVI / eSSVI surface formulas.

Gatheral & Jacquier (2014) SSVI builds a surface thats arbitrage-free
by construction when the angle function ``psi(theta)`` satisfies the
Gatheral-Jacquier condition.  The eSSVI specialization uses
``psi = eta / theta**gamma`` — a power-law decay thats safe for
``0 <= gamma <= 1, eta > 0``.
"""

from math import sqrt
from dataclasses import dataclass

from pydantic import BaseModel, Field


class SSVIParams(BaseModel):
    """SSVI parameters for one expiry slice.

    - ``theta``: ATM total variance (sigma_atm**2 * T)
    - ``rho``:   correlation between spot and vol, -1 < rho < 1
    - ``psi``:   angle function at this theta (controls wing slope)
    """
    theta: float= Field(..., gt=0)
    rho: float= Field(..., gt=-1, lt=1)
    psi: float= Field(..., gt=0)


class eSSVISurfaceParams(BaseModel):
    """eSSVI wing function: psi(theta)= eta / theta**gamma.

    - ``eta``:   power-law coefficient, > 0
    - ``gamma``: power-law exponent, in [0, 1] for arb-free surfaces
    """
    eta: float= Field(..., gt=0)
    gamma: float= Field(..., ge=0, le=1)


def essvi_psi(theta: float, eta: float, gamma: float) -> float:
    """eSSVI angle function: psi = eta / theta**gamma."""
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")
    return eta / (theta ** gamma)


def ssvi_w(k: float, theta: float, rho: float, psi: float) -> float:
    """SSVI total variance at log-moneyness ``k``.

    The standard formula:
        w = (theta / 2) * (1 + rho*psi*k + sqrt((psi*k + rho)**2 + (1 - rho**2)))
    """
    return (theta / 2.0) * (1.0 + rho * psi * k + sqrt((psi * k + rho) ** 2 + (1.0 - rho ** 2)))


def essvi_w(k: float, theta: float, rho: float, eta: float, gamma: float) -> float:
    """eSSVI total variance: SSVI with psi set to eta / theta**gamma."""
    return ssvi_w(k, theta, rho, essvi_psi(theta, eta, gamma))


def ssvi_dw_dk(k: float, theta: float, rho: float, psi: float) -> float:
    """First derivative of SSVI w.r.t. k (smile slope)."""
    pk= psi * k + rho
    inner= pk * pk + (1.0 - rho * rho)
    return (theta / 2.0) * (rho * psi + psi * pk / sqrt(inner))


def ssvi_d2w_dk2(k: float, theta: float, rho: float, psi: float) -> float:
    """Second derivative of SSVI w.r.t. k (smile curvature)."""
    pk= psi * k + rho
    inner= pk * pk + (1.0 - rho * rho)
    return (theta / 2.0) * (psi * psi * (1.0 - rho * rho) / (inner ** 1.5))


def gatheral_jacquier_condition(theta: float, rho: float, psi: float) -> float:
    """Sufficient no-arb condition for SSVI (Gatheral & Jacquier 2014).

    From GJ (2014) Theorem 4.2: a slice is free of butterfly arbitrage
    when ``theta * psi * (1 + |rho|) <= 4``.

    Returns the residual ``4.0 - theta * psi * (1.0 + abs(rho))``.
    The slice is arb-free when the residual >= 0.
    If ``|rho| >= 1.0``, returns ``float('-inf')`` (always a violation).

    Reference
    ---------
    Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
    Quantitative Finance, 14(1), 59-71.
    """
    abs_rho = abs(rho)
    if abs_rho >= 1.0:
        return float("-inf")
    return 4.0 - theta * psi * (1.0 + abs_rho)


def essvi_arb_safe(theta: float, eta: float, gamma: float) -> bool:
    """Quick check that eSSVI power-law parameters are in the valid range.

    Returns ``True`` when ``0 <= gamma <= 1`` and ``eta > 0``.
    This checks the necessary structural bounds on the eSSVI wing
    function but does NOT verify the full Gatheral-Jacquier condition
    ``theta * psi * (1+|rho|) <= 4`` — that requires evaluating every
    slice's (theta, rho) pair against the wing function.  Use
    :func:`gatheral_jacquier_condition` or
    :func:`arbfree_vol.arbitrage.svi_detect.detect_svi_surface` for a
    complete no-arbitrage check.
    """
    return 0.0 <= gamma <= 1.0 and eta > 0.0


def to_raw_svi_params(theta: float, rho: float, psi: float) -> tuple[float, float, float, float, float]:
    """Convert SSVI (theta, rho, psi) into raw SVI (a, b, rho, m, sigma).

    The mapping is exact across all k:
        b   = theta * psi / 2
        m   = -rho / psi
        sigma = sqrt(1 - rho**2) / psi
        a   = (theta / 2) * (1 - rho**2)
        rho passed through unchanged

    This lets us reuse the existing SVI pipeline (detection, plots)
    on an SSVI-fitted surface.
    """
    b = theta * psi / 2.0
    if psi <= 0:
        raise ValueError("psi must be positive")
    m = -rho / psi
    sigma = (1.0 - rho * rho) ** 0.5 / psi
    a = (theta / 2.0) * (1.0 - rho * rho)
    return a, b, rho, m, sigma
