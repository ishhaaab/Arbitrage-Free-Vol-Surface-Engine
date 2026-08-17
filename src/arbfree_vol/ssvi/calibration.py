"""Calibration: fit SSVI / eSSVI to observed (k, w) points."""

import numpy as np
from scipy.optimize import least_squares

from arbfree_vol.ssvi.model import SSVIParams, ssvi_w


def fit_ssvi_slice(points: list[tuple[float, float]]) -> SSVIParams:
    """Fit SSVI (theta, rho, psi) to a single slice of (k, w) points.

    Uses scipy least_squares with bounds to keep rho in (-1, 1)
    and theta, psi positive.

    Returns the fitted SSVIParams.  Raises ValueError on too few points.
    """
    if len(points) < 5:
        raise ValueError("Need at least 5 points to fit SSVI slice")

    # initial guess: theta= min(w) (ATM variance= min total var),
    # rho= 0 (no skew), psi= 0.5
    ws= np.array([w for _, w in points])
    w_min= float(np.min(ws))
    x0= [w_min, 0.0, 0.5]
    bounds= (
        [1e-6, -0.999, 1e-6],
        [10.0, 0.999, 20.0],
    )

    def residuals(p):
        theta, rho, psi= p
        return [ssvi_w(float(k), theta, rho, psi) - float(w) for k, w in points]

    result= least_squares(residuals, x0, bounds=bounds)
    if not result.success:
        raise RuntimeError(f"SSVI calibration failed: {result.message}")
    theta, rho, psi= result.x
    return SSVIParams(theta=float(theta), rho=float(rho), psi=float(psi))

