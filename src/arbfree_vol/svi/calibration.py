from math import sqrt

from arbfree_vol.svi.model import SVIParams, svi_total_variance, svi_g
from arbfree_vol.svi.data import slice_to_point
from arbfree_vol.models.surface import VolSurface, ExpirySlice


from scipy.optimize import least_squares
import numpy as np


def _min_total_variance(a: float, b: float, rho: float, sigma: float) -> float:
    """Minimum total variance of the SVI curve: a + b * sigma * sqrt(1 - rho^2)."""
    return a + b * sigma * sqrt(1.0 - rho * rho)


def calibrate(points: list[tuple[float,float]]) -> SVIParams:
    """Fit raw SVI parameters (a, b, rho, m, sigma) to (k, w) points."""
    if len(points) < 5:
        raise ValueError("need at least 5 points to fit SVI")

    def residuals(p):
        """Return model minus market total variance at each (k, w)."""
        return [svi_total_variance(k, *p) - w for k, w in points]

    x0 = [min(w[1] for w in points), 0.1, -0.5, 0.0, 0.1]

    bounds = ([-np.inf, 0, -0.999, -np.inf, 1e-6], [np.inf, np.inf, 0.999, np.inf, np.inf])

    result = least_squares(residuals, x0, bounds=bounds)
    if not result.success:
        raise RuntimeError(f"SVI calibration failed: {result.message}")
    a, b, rho, m, sigma = result.x

    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def calibrate_constrained(
    points: list[tuple[float, float]],
    arb_penalty: float = 100.0,
    k_min: float = -3.0,
    k_max: float = 3.0,
    n_k: int = 121,
) -> SVIParams:
    r"""Fit raw SVI with a smooth penalty on butterfly arbitrage (g(k) < 0).

    Augments the data residual vector with:

    * ``sqrt(arb_penalty) * sqrt(max(-g(k_j), 0))`` for each point on a
      uniform k-grid covering [k_min, k_max] (``n_k`` points) — this
      penalises negative risk-neutral density.
    * ``sqrt(arb_penalty) * sqrt(max(-min_total_variance, 0))`` — this
      penalises negative minimum total variance.

    When all constraints are satisfied the penalty residuals are zero
    and the fit reduces to the standard ``calibrate()`` (modulo
    optimizer path).
    """
    if len(points) < 5:
        raise ValueError("need at least 5 points to fit SVI")

    def residuals(p):
        a, b, rho, m, sigma = p

        # ----- data fit -----
        data_res = [svi_total_variance(k, a, b, rho, m, sigma) - w for k, w in points]

        # ----- butterfly (g(k) >= 0) penalty on a fixed k-grid -----
        k_grid = np.linspace(k_min, k_max, n_k)
        sqrt_pen = sqrt(arb_penalty)
        arb_res = [
            sqrt_pen * sqrt(max(-svi_g(k, a, b, rho, m, sigma), 0.0))
            for k in k_grid
        ]

        # ----- min-variance penalty (w_min >= 0) -----
        w_min = _min_total_variance(a, b, rho, sigma)
        min_var_res = [sqrt_pen * sqrt(max(-w_min, 0.0))]

        return data_res + arb_res + min_var_res

    x0 = [min(w[1] for w in points), 0.1, -0.5, 0.0, 0.1]
    bounds = ([-np.inf, 0, -0.999, -np.inf, 1e-6], [np.inf, np.inf, 0.999, np.inf, np.inf])

    result = least_squares(residuals, x0, bounds=bounds)
    if not result.success:
        raise RuntimeError(f"SVI constrained calibration failed: {result.message}")
    a, b, rho, m, sigma = result.x
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def calibrate_slice(surface: VolSurface, s: ExpirySlice) -> SVIParams:
    """Convenience: calls ``slice_to_point`` then ``calibrate``."""
    return calibrate(slice_to_point(surface, s))
