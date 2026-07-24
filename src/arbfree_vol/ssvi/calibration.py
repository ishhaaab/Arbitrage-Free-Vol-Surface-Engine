"""Calibration: fit SSVI / eSSVI to observed (k, w) points."""

from math import sqrt, log
from statistics import mean

import numpy as np
from scipy.optimize import least_squares

from arbfree_vol.ssvi.model import (
    SSVIParams,
    eSSVISurfaceParams,
    essvi_psi,
    ssvi_w,
    essvi_w,
)


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
    ks= np.array([k for k, _ in points])
    ws= np.array([w for _, w in points])
    w_min= float(np.min(ws))
    w_max= float(np.max(ws))
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


def fit_essvi_slice(
        points: list[tuple[float, float]]
        ) -> tuple[SSVIParams, eSSVISurfaceParams]:
    """Fit eSSVI surface to a single slice.

    Fits both the per-slice (theta, rho) and the (eta, gamma) power-law
    wing function jointly.  Note: fitting the wing function from a
    single slice is constrained; in practice, fit across multiple
    slices.  This routine is a starting point for a per-slice fit.
    """
    if len(points) < 5:
        raise ValueError("Need at least 5 points to fit eSSVI slice")

    # Initial guess: psi(theta)= eta / theta ** gamma.  Start with gamma=0
    # (constant psi) and eta = some moderate value.
    ks= np.array([k for k, _ in points])
    ws= np.array([w for _, w in points])
    theta0= float(np.min(ws))
    rho0= 0.0
    eta0= 0.5
    gamma0= 0.5
    x0= [theta0, rho0, eta0, gamma0]
    bounds= (
        [1e-6, -0.999, 1e-6, 0.0],
        [10.0, 0.999, 20.0, 1.0],
    )

    def residuals(p):
        theta, rho, eta, gamma= p
        return [
            essvi_w(float(k), theta, rho, eta, gamma) - float(w)
            for k, w in points
        ]

    result= least_squares(residuals, x0, bounds=bounds)
    if not result.success:
        raise RuntimeError(f"eSSVI calibration failed: {result.message}")
    theta, rho, eta, gamma= result.x
    return (
        SSVIParams(theta=float(theta), rho=float(rho), psi=essvi_psi(theta, eta, gamma)),
        eSSVISurfaceParams(eta=float(eta), gamma=float(gamma)),
    )
