from collections.abc import Callable
from math import sqrt

from arbfree_vol.svi.model import SVIParams, svi_total_variance, svi_g


from scipy.optimize import least_squares
import numpy as np


# Evaluation budget for the unconstrained warm-start fit used by the
# constrained multi-start.  Empirically measured (2026-08-08, scipy
# 1.17.1): clean synthetic SVI data needs only 9 nfev (T1 fixture,
# 15 strikes) / 116 nfev (Gatheral 2004 fit tuple, 19 strikes; stable
# across 10 repeated runs), while the largest real SPX slice (545
# strikes) needs ~10631 nfev to converge.  The cap must sit above the
# clean-data requirement so the warm start still converges on
# clean/synthetic data (that is the whole point of the multi-start), yet
# small enough that on real data it fails FAST instead of burning
# scipy's default ~500-eval full failure (~1.1s per slice).  150 sits
# 1.3x above the most demanding clean case while cutting the
# real-fixture warm-start waste from ~8.5s (default budget) to ~2.6s
# across the 7-slice SPX fixture, bringing repair() wall time to ~1.4x
# of the pre-multi-start code (measured 7.37s vs 5.28s median).
_WARM_START_MAX_NFEV = 150


def _min_total_variance(a: float, b: float, rho: float, sigma: float) -> float:
    """Minimum total variance of the SVI curve: a + b * sigma * sqrt(1 - rho^2)."""
    return a + b * sigma * sqrt(1.0 - rho * rho)


def _penalty_term(pen: float, value: float) -> float:
    """One smooth-penalty residual: ``sqrt(arb_penalty) * sqrt(max(value, 0))``."""
    return pen * sqrt(max(value, 0.0))


def _build_residuals(
    points: list[tuple[float, float]],
    k_grid: np.ndarray,
    sqrt_pen: float,
    prev_slice: SVIParams | None,
) -> Callable[[np.ndarray], list[float]]:
    """Return the augmented residual vector for constrained SVI fitting.

    Concatenates the data-fit residuals with three penalty groups:
    butterfly (``g(k) < 0``), minimum total variance (``< 0``), and —
    when ``prev_slice`` is supplied — calendar (``w(k) < w_prev(k)``).
    Each penalty term is zero whenever its constraint holds, so a fully
    feasible fit reduces to the standard ``calibrate()`` objective.
    """
    def residuals(p):
        a, b, rho, m, sigma = p

        # ----- data fit -----
        data_res = [svi_total_variance(k, a, b, rho, m, sigma) - w for k, w in points]

        # ----- butterfly (g(k) >= 0) penalty on the fixed k-grid -----
        arb_res = [
            _penalty_term(sqrt_pen, -svi_g(k, a, b, rho, m, sigma))
            for k in k_grid
        ]

        # ----- min-variance penalty (w_min >= 0) -----
        w_min = _min_total_variance(a, b, rho, sigma)
        min_var_res = [_penalty_term(sqrt_pen, -w_min)]

        # ----- calendar penalty: w(k) >= w_prev(k) for all k on grid -----
        # Only added when a previous fitted slice is supplied.  Penalises
        # the current slice's total variance dipping below the previous
        # (shorter-T) slice's at any k, which is calendar arbitrage.
        if prev_slice is not None:
            cal_res = [
                _penalty_term(
                    sqrt_pen,
                    svi_total_variance(k, prev_slice.a, prev_slice.b,
                                       prev_slice.rho, prev_slice.m,
                                       prev_slice.sigma)
                    - svi_total_variance(k, a, b, rho, m, sigma),
                )
                for k in k_grid
            ]
        else:
            cal_res = []

        return data_res + arb_res + min_var_res + cal_res

    return residuals


def _run_multistart(
    residuals: Callable[[np.ndarray], list[float]],
    starts: list[list[float]],
    bounds,
):
    """Run ``least_squares`` from each start and return the best result.

    Only successful, finite-cost runs are eligible: a failed run can
    otherwise win on a spuriously low residual cost and mask a valid
    successful start.  A start whose residual vector is non-finite at its
    initial point makes ``least_squares`` raise ``ValueError``; that start
    is treated as a failed start (never crashing the repair pipeline),
    mirroring the ``RuntimeError`` path.  Returns ``(best, last_failure)``
    where ``best`` is ``None`` if every start failed.
    """
    best = None
    best_cost = float("inf")
    last_failure = None
    for x0 in starts:
        try:
            result = least_squares(residuals, x0, bounds=bounds, max_nfev=5000)
        except ValueError as exc:
            # An unevaluable start (non-finite residuals at x0) is a
            # failed start, not a crash — live SPY produced extreme
            # warm-start params (b ~ 30, rho ~ 0.997) that triggered this.
            last_failure = exc
            continue
        if not result.success:
            last_failure = result
            continue
        cost = float(np.sum(np.square(result.fun)))
        if np.isfinite(cost) and cost < best_cost:
            best_cost = cost
            best = result
    return best, last_failure


def calibrate(points: list[tuple[float,float]],
              max_nfev: int | None = None) -> SVIParams:
    """Fit raw SVI parameters (a, b, rho, m, sigma) to (k, w) points.

    ``max_nfev`` is threaded through to the internal ``least_squares``
    call; ``None`` keeps scipy's default budget.
    """
    if len(points) < 5:
        raise ValueError("need at least 5 points to fit SVI")

    def residuals(p):
        """Return model minus market total variance at each (k, w)."""
        return [svi_total_variance(k, *p) - w for k, w in points]

    x0 = [min(w[1] for w in points), 0.1, -0.5, 0.0, 0.1]

    bounds = ([-np.inf, 0, -0.999, -np.inf, 1e-6], [np.inf, np.inf, 0.999, np.inf, np.inf])

    result = least_squares(residuals, x0, bounds=bounds, max_nfev=max_nfev)
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

    k_grid = np.linspace(k_min, k_max, n_k)
    sqrt_pen = sqrt(arb_penalty)
    residuals = _build_residuals(points, k_grid, sqrt_pen, prev_slice)

    bounds = ([-np.inf, 0, -0.999, -np.inf, 1e-6], [np.inf, np.inf, 0.999, np.inf, np.inf])

    # Multi-start: the fixed default seed (a=min(w), b=0.1, rho=-0.5,
    # m=0.0, sigma=0.1) can stall in a non-smooth local minimum of the
    # penalty landscape — observed on clean SVI data whose true m is far
    # from 0 (e.g. the Gatheral 2004 fit tuple m=-0.569) and on the
    # calendar-penalty path where w(k) == w_prev(k) puts the true solution
    # on the penalty kink.  We ALSO warm-start from the unconstrained fit —
    # whenever the constraints are (nearly) satisfied the constrained
    # optimum sits near it — and keep the lower-cost result.  The
    # unconstrained seed is skipped only if it itself fails.
    starts = [[min(w[1] for w in points), 0.1, -0.5, 0.0, 0.1]]
    try:
        # Warm-start from the unconstrained fit under a capped evaluation
        # budget: clean/synthetic data converges well inside
        # _WARM_START_MAX_NFEV (see the constant comment), while real
        # 100+ point slices would exhaust even scipy's default budget and
        # burn ~1.1s per slice before failing — the cap makes that failure
        # fast and cheap.
        unconstrained = calibrate(points, max_nfev=_WARM_START_MAX_NFEV)
        starts.append([unconstrained.a, unconstrained.b, unconstrained.rho,
                       unconstrained.m, unconstrained.sigma])
    except (ValueError, RuntimeError):
        # Warm start is best-effort: skip it if the unconstrained fit
        # fails, rather than aborting the constrained multi-start.
        pass

    best, last_failure = _run_multistart(residuals, starts, bounds)

    if best is None:
        # All starts failed — report the message from the first failed
        # start (if any) or a generic failure.
        message = (
            getattr(last_failure, "message", None) or str(last_failure)
            if last_failure is not None else "no start"
        )
        raise RuntimeError(f"SVI constrained calibration failed: {message}")
    a, b, rho, m, sigma = best.x
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)
