from math import sqrt

from arbfree_vol.svi.model import SVIParams, svi_total_variance, svi_g


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
    prev_slice: SVIParams | None = None,
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

    * ``sqrt(arb_penalty) * sqrt(max(w_prev(k_j) - w(k_j), 0))`` for each
      point on the same k-grid — this penalises calendar arbitrage.  It
      activates only when *prev_slice* is provided (i.e. a shorter-dated
      slice has already been fitted).  The condition ``w(k) >= w_prev(k)``
      is the calendar no-arbitrage condition (total variance non-decreasing
      in maturity T).  Uses the same *arb_penalty* weight as the other
      penalty terms.
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

        # ----- calendar penalty: w(k) >= w_prev(k) for all k on grid -----
        # Only added when a previous fitted slice is supplied.  Penalises
        # the current slice's total variance dipping below the previous
        # (shorter-T) slice's at any k, which is calendar arbitrage.
        if prev_slice is not None:
            cal_res = [
                sqrt_pen * sqrt(max(
                    svi_total_variance(k, prev_slice.a, prev_slice.b,
                                       prev_slice.rho, prev_slice.m,
                                       prev_slice.sigma)
                    - svi_total_variance(k, a, b, rho, m, sigma),
                    0.0,
                ))
                for k in k_grid
            ]
        else:
            cal_res = []

        return data_res + arb_res + min_var_res + cal_res

    x0 = [min(w[1] for w in points), 0.1, -0.5, 0.0, 0.1]
    bounds = ([-np.inf, 0, -0.999, -np.inf, 1e-6], [np.inf, np.inf, 0.999, np.inf, np.inf])

    result = least_squares(residuals, x0, bounds=bounds, max_nfev=5000)
    if not result.success:
        raise RuntimeError(f"SVI constrained calibration failed: {result.message}")
    a, b, rho, m, sigma = result.x
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
