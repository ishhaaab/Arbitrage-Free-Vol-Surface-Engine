"""SABR model (Hagan et al. 2002) — asymptotic Black implied volatility.

Provides the standard Hagan asymptotic implied vol formula, the SABRParams
boundary type, and an adapter ``to_raw_svi_params`` that maps a SABR smile
to a best-fitting raw SVI parameter set.

Limitations
-----------
The SABR implied vol formula is an asymptotic expansion; it is accurate for
short to moderate maturities and strikes not too far from the forward.  The
SABR smile is *not* guaranteed to be arbitrage-free — only the "nearly
arbitrage-free" asymptotic regime holds.  The repair pipeline runs
``detect_svi_surface`` on the mapped SVI parameters and reports any violations
truthfully.
"""

from math import exp, log, sqrt

import numpy as np
from pydantic import BaseModel, Field


class SABRParams(BaseModel):
    """SABR model parameters for one expiry slice.

    The model: dF = alpha * F^beta * dW1,  d(ln alpha) = nu * dW2,
    with dW1 * dW2 = rho * dt.

    Parameters
    ----------
    alpha : initial ATM vol (> 0)
    beta  : shape parameter in [0, 1] (0 = stochastic-normal, 1 = lognormal).
            Default 0.5 (common for equities).
    rho   : correlation between spot and vol, -1 < rho < 1.
    nu    : vol-of-vol (> 0).
    """
    alpha: float = Field(..., gt=0)
    beta: float = Field(0.5, ge=0, le=1)
    rho: float = Field(..., gt=-0.999, lt=0.999)
    nu: float = Field(..., gt=0)


_SABR_NEAR_ATM_EPS = 1e-8
"""Threshold below which we use the ATM limit formula."""


def _sabr_correction(FK_pow: float, FK_1mb: float,
                     alpha: float, beta: float, rho: float, nu: float) -> float:
    """Common time-correction term in the Hagan et al. formula (bracketed T term)."""
    return (
        ((1.0 - beta) ** 2 / 24.0) * alpha ** 2 / FK_1mb
        + (rho * beta * alpha * nu) / (4.0 * FK_pow)
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    )


def sabr_implied_vol(k: float, F: float, T: float,
                     alpha: float, beta: float, rho: float, nu: float) -> float:
    """SABR asymptotic Black implied volatility (Hagan et al. 2002).

    Uses the standard Hagan-Kumar-Lesniewski-Woodward formula with the
    full T-correction.  For ``|k| <= 1e-8`` the ATM closed-form limit is
    used to avoid the degenerate 0/0 in ``z / x(z)``.

    Parameters
    ----------
    k : log-moneyness ``ln(K/F)``
    F : forward price
    T : time to expiry (years)
    alpha, beta, rho, nu : SABR model parameters

    Returns
    -------
    sigma_B : Black implied volatility.

    Reference
    ---------
    Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).
    Managing Smile Risk.  Wilmott Magazine, 1, 84-108.
    """
    if abs(k) < _SABR_NEAR_ATM_EPS:
        # ATM limit: K = F, FK_pow = F^(1-beta), FK_1mb = F^(2-2*beta)
        F_1mb = F ** (1.0 - beta)
        F_2mb = F ** (2.0 - 2.0 * beta)
        sigma_atm = alpha / F_1mb
        corr = _sabr_correction(F_1mb, F_2mb, alpha, beta, rho, nu)
        return sigma_atm * (1.0 + corr * T)

    K = F * exp(k)
    FK = F * K
    FK_pow = FK ** ((1.0 - beta) / 2.0)
    FK_1mb = FK ** (1.0 - beta)

    # z = (nu / alpha) * (F * K)^((1-beta)/2) * ln(F/K)
    # Note: ln(F/K) = -k since k = ln(K/F)
    z = (nu / alpha) * FK_pow * log(F / K)

    # x(z) = ln( (D + z - rho) / (1 - rho) ),  D = sqrt(1 - 2*rho*z + z^2)
    if abs(z) < _SABR_NEAR_ATM_EPS:
        z_over_x = 1.0  # limit z -> 0
    else:
        D = sqrt(1.0 - 2.0 * rho * z + z * z)
        x_chi = log((D + z - rho) / (1.0 - rho))
        z_over_x = z / x_chi

    sigma_base = alpha / FK_pow * z_over_x
    corr = _sabr_correction(FK_pow, FK_1mb, alpha, beta, rho, nu)

    return sigma_base * (1.0 + corr * T)


def sabr_total_variance(k: float, F: float, T: float,
                        alpha: float, beta: float, rho: float, nu: float) -> float:
    """SABR total variance ``w(k) = sigma_B(k)^2 * T``.

    Float-level pure function (no Pydantic).  Useful as a drop-in for the
    total-variance interface expected by calibration and detection routines.
    """
    sigma = sabr_implied_vol(k, F, T, alpha, beta, rho, nu)
    return sigma * sigma * T


def to_raw_svi_params(sabr_params: SABRParams,
                       forward: float,
                       expiry_time: float,
                       k_grid: np.ndarray | None = None) -> tuple[float, float, float, float, float]:
    """Map SABR smile to best-fitting raw SVI (a, b, rho, m, sigma).

    Samples ``w_sabr(k) = sabr_total_variance(k, forward, expiry_time, ...)``
    on a dense k-grid (default ``np.linspace(-3, 3, 241)``), then runs
    ``scipy.optimize.least_squares`` to fit raw SVI parameters.

    .. note::
       SABR is not exactly SVI-representable.  The fit residual measures
       the mapping error; the caller should inspect ``rmse`` if accuracy
       is critical.

    Returns
    -------
    tuple[float, float, float, float, float]
        Fitted (a, b, rho, m, sigma) parameters.
    """
    from scipy.optimize import least_squares
    from arbfree_vol.svi.model import svi_total_variance

    if k_grid is None:
        k_grid = np.linspace(-3.0, 3.0, 241)

    alpha = sabr_params.alpha
    beta = sabr_params.beta
    rho = sabr_params.rho
    nu = sabr_params.nu

    # Sample SABR total variance on the k-grid
    w_target = np.array([
        sabr_total_variance(float(k), forward, expiry_time, alpha, beta, rho, nu)
        for k in k_grid
    ])

    # Initial guess from ATM-like parameters
    w0 = float(w_target[len(k_grid) // 2])  # w at k≈0
    x0 = [w0 * 0.5, 0.2, 0.0, 0.0, 0.3]

    bounds = (
        [-np.inf, 0.0, -0.999, -np.inf, 1e-6],
        [np.inf, np.inf, 0.999, np.inf, np.inf],
    )

    def residuals(p):
        a, b, r, m, s = p
        return [svi_total_variance(float(k), a, b, r, m, s) - float(w)
                for k, w in zip(k_grid, w_target)]

    result = least_squares(residuals, x0, bounds=bounds)
    if not result.success:
        raise RuntimeError(f"SABR-to-SVI mapping failed: {result.message}")
    return tuple(result.x)  # type: ignore[return-value]
