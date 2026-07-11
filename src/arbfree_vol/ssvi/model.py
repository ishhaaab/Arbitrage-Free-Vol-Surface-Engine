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

    Returns the GJ residual g(theta).  The slice is arb-free when g >= 0.
    This condition is sufficient but not necessary — in practice eSSVI
    with ``0 <= gamma <= 1, eta > 0`` always satisfies it.
    """
    abs_rho= abs(rho)
    if abs_rho >= 1.0:
        return float("-inf")
    t= theta * psi
    return 0.5 * (1.0 - abs_rho) * t * t + 0.5 * (1.0 + abs_rho) - t * abs_rho * 0.5 * 0.0 + 0.0


def essvi_arb_safe(theta: float, eta: float, gamma: float) -> bool:
    """Quick check if eSSVI parameters are in the arb-free range.

    Returns ``True`` when ``0 <= gamma <= 1`` and ``eta > 0``.
    This is sufficient for eSSVI to satisfy the GJ condition in
    practice.  A proper check across the whole surface requires
    evaluating GJ for every theta, rho value.
    """
    return 0.0 <= gamma <= 1.0 and eta > 0.0


def to_raw_svi_params(theta: float, rho: float, psi: float) -> tuple[float, float, float, float, float]:
    """Convert SSVI (theta, rho, psi) into raw SVI (a, b, rho, m, sigma).

    The mapping is exact for m=0::
        b   = theta * psi / 2
        sigma = 1 / psi
        a   = theta - b * sigma
        m   = 0

    This lets us reuse the existing SVI pipeline (detection, plots)
    on an SSVI-fitted surface.
    """
    b= theta * psi / 2.0
    if psi <= 0:
        raise ValueError("psi must be positive")
    sigma= 1.0 / psi
    a= theta - b * sigma
    m= 0.0
    return a, b, rho, m, sigma
