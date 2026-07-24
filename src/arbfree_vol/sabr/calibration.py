"""Calibration: fit SABR (alpha, rho, nu) to observed (k, w) points at fixed beta."""

from math import sqrt
from statistics import mean

import numpy as np
from scipy.optimize import least_squares

from arbfree_vol.sabr.model import SABRParams, sabr_total_variance


def calibrate_sabr(points: list[tuple[float, float]],
                   forward: float,
                   expiry_time: float,
                   beta_hint: float = 0.5) -> SABRParams:
    """Fit SABR (alpha, rho, nu) to (k, w) points at a fixed beta.

    Parameters
    ----------
    points : list of (k, w) tuples
        Log-moneyness and total variance observations.
    forward : float
        Forward price for the slice.
    expiry_time : float
        Time to expiry in years.
    beta_hint : float, optional
        Fixed beta parameter (default 0.5, common for equities).
        Calibrated separately or set a-priori per market convention.

    Returns
    -------
    SABRParams with the fitted (alpha, rho, nu) and the fixed beta_hint.

    Raises
    ------
    ValueError
        If fewer than 5 points are provided.
    """
    if len(points) < 5:
        raise ValueError(f"Need at least 5 points to calibrate SABR, got {len(points)}")

    ks = np.array([p[0] for p in points])
    ws = np.array([p[1] for p in points])

    # Initial guess: use ATM total variance for alpha, moderate nu, zero skew
    # Find w at k≈0 by taking the point nearest to zero
    idx_atm = int(np.argmin(np.abs(ks)))
    w_atm = float(ws[idx_atm])

    alpha0 = sqrt(w_atm / expiry_time) if expiry_time > 0 else 0.2
    x0 = [alpha0, 0.0, 0.3]

    bounds = (
        [1e-6, -0.999, 1e-6],
        [10.0, 0.999, 10.0],
    )

    def residuals(p):
        alpha, rho, nu = p
        return [
            sabr_total_variance(float(k), forward, expiry_time,
                                alpha, beta_hint, rho, nu) - float(w)
            for k, w in points
        ]

    result = least_squares(residuals, x0, bounds=bounds)
    if not result.success:
        raise RuntimeError(f"SABR calibration failed: {result.message}")
    alpha, rho, nu = result.x

    return SABRParams(alpha=float(alpha), beta=beta_hint,
                      rho=float(rho), nu=float(nu))
