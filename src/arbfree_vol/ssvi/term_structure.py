"""Calendar-arbitrage-free eSSVI term-structure calibration.

Fits SSVI slices sequentially by increasing maturity, enforcing the
Hendriks & Martini (2019) Prop 3.1 no-calendar-spread condition as
HARD optimizer constraints.  Per-slice rho stays fully free
(tanh-reparametrised, no cross-slice functional form).

The discrete sequential calibration procedure follows
Corbetta, Cohort, Laachir & Martini (2019), Sec 2.2-2.3
(arXiv:1804.04924).

Conditions enforced
-------------------
For slices ordered by increasing maturity T_1 < T_2 < ... < T_N,
let  p_i  = SSVIParams.psi  (the angle / wing function),
     theta_i = SSVIParams.theta,
     rho_i   = SSVIParams.rho,
     chi_i   = theta_i * p_i.

The surface is free of calendar-spread arbitrage iff (Prop 3.1):

  (a) theta_1 <= theta_2 <= ... <= theta_N           (non-decreasing ATM variance)
  (b) chi_1   <= chi_2   <= ... <= chi_N             (non-decreasing wing magnitude)
  (c) for each adjacent pair (i, i+1):
      | (rho_{i+1}*chi_{i+1} - rho_i*chi_i) / (chi_{i+1} - chi_i) | <= 1

Butterfly arbitrage per slice (Gatheral & Jacquier 2014, Theorem 4.2):
  theta * psi * (1 + |rho|) < 4   (STRICT; enforced with a small margin)
  theta * psi^2 * (1 + |rho|) <= 4   (non-strict)
Both are written as smooth pairs of inequalities using (1+rho) and (1-rho).

References
----------
.. [HM19] Hendriks, S. & Martini, C. (2019). "The Extended SSVI
   Volatility Surface", J. Comput. Finance 22(5), 25-39.  Prop 3.1.
.. [CCLM19] Corbetta, J., Cohort, P., Laachir, I. & Martini, C.
   (2019). "Robust calibration and arbitrage-free interpolation of
   SSVI slices", arXiv:1804.04924, Sec 2.2-2.3.
.. [GJ14] Gatheral, J. & Jacquier, A. (2014). "Arbitrage-free SVI
   volatility surfaces", Quant. Finance 14(1), 59-71.

Module layout
-------------
This file holds the sequential fit pipeline and remains the public
namespace; butterfly constraints live in ``_butterfly``, H&M margin
helpers in ``_hm_margin``, post-fit verifiers in ``_hm_verify``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, Bounds

from arbfree_vol.ssvi.model import SSVIParams, ssvi_w
from arbfree_vol.ssvi.calibration import fit_ssvi_slice
from arbfree_vol.ssvi._constraints import _hard_constraints, _constrained_minimize

# Re-exports preserving the pre-split import surface: the pure leaf
# helpers now live in the private sibling modules below, and importing
# them from ``term_structure`` keeps every existing ``from
# arbfree_vol.ssvi.term_structure import ...`` call site working
# unchanged.  Declared in ``__all__`` so linters treat them as the
# module's public namespace rather than unused imports.
from arbfree_vol.ssvi._butterfly import _GJ_CONDITION1_STRICT_EPS, _butterfly_constraints
from arbfree_vol.ssvi._hm_margin import (
    _EPS_THETA,
    _EPS_CHI,
    _HM_BOUNDARY_MARGIN_THETA,
    _HM_BOUNDARY_MARGIN_CHI,
    _HM_BOUNDARY_MARGIN_RATIO,
    _HM_RMSE_RATIO_MAX,
    _HM_RMSE_FLOOR,
    _slice_rmse,
    _hm_boundary_deltas,
    _within_boundary_window,
)
from arbfree_vol.ssvi._hm_verify import (
    verify_hm_condition,
    verify_ssvi_calendar_free,
    verify_hm_condition_breakdown,
    _hm_breakdown_entry,
)

# Public namespace: the sequential fit API plus the re-exported leaf
# helpers above (``from arbfree_vol.ssvi.term_structure import *``
# yields exactly these names).
__all__ = [
    "SequentialFitResult",
    "fit_ssvi_surface_sequential",
    "_GJ_CONDITION1_STRICT_EPS",
    "_butterfly_constraints",
    "_EPS_THETA",
    "_EPS_CHI",
    "_HM_BOUNDARY_MARGIN_THETA",
    "_HM_BOUNDARY_MARGIN_CHI",
    "_HM_BOUNDARY_MARGIN_RATIO",
    "_HM_RMSE_RATIO_MAX",
    "_HM_RMSE_FLOOR",
    "_slice_rmse",
    "_hm_boundary_deltas",
    "_within_boundary_window",
    "verify_hm_condition",
    "verify_ssvi_calendar_free",
    "verify_hm_condition_breakdown",
    "_hm_breakdown_entry",
]

_logger = logging.getLogger(__name__)


@dataclass
class SequentialFitResult:
    """Result of a sequential eSSVI term-structure fit.

    Attributes
    ----------
    fitted_slices : list of (expiry_time, SSVIParams)
        Successful fits — either hard-constrained (H&M Prop 3.1) or
        unconstrained fallback.  Slices where *both* fits failed are
        absent.
    fallback_slices : list of float
        Expiry times where the hard-constrained fit failed but the
        unconstrained per-slice fallback succeeded.  These slices are
        NOT arbitrage-free by construction.
    failed_slices : list of float
        Expiry times where both the hard-constrained and the
        unconstrained fit failed.  These slices are omitted from
        ``fitted_slices`` entirely.
    fitted_slices_prev : list of float | None
        For each entry in ``fitted_slices``, the expiry time of the
        actual ``prev`` slice used in the calibration.  The first
        fitted slice has ``None`` (no predecessor).  For a fallback
        slice, this records the T of the last *hard-constrained* fit
        before it — which may be several positions back in the
        fitted sequence when there are consecutive fallbacks.
    """
    fitted_slices: list[tuple[float, SSVIParams]]
    fallback_slices: list[float]
    failed_slices: list[float]
    fitted_slices_prev: list[float | None] = field(default_factory=list)


def _initial_guess(
    points: list[tuple[float, float]],
    prev: SSVIParams | None,
    eps_theta: float,
    eps_chi: float,
) -> NDArray[np.float64]:
    """Seed ``x0 = (theta, arctanh(rho), log(psi))`` for the constrained fit.

    Least-squares seed in (theta, rho, psi) space, adjusted for the
    Hendriks-Martini calendar constraints when a predecessor slice is
    given, then mapped to the optimizer's unconstrained parameterization.
    """
    ws = np.array([w for _, w in points])
    from scipy.optimize import least_squares as _ls

    def _seed_resid(p):
        th, rh, ps = p
        return np.array([
            ssvi_w(float(k), th, rh, ps) - float(w)
            for k, w in points
        ])

    seed_result = _ls(
        _seed_resid,
        x0=[float(np.min(ws)), 0.0, 0.5],
        bounds=([1e-6, -0.999, 1e-6], [10.0, 0.999, 20.0]),
    )
    theta0, rho0, p0 = [float(v) for v in seed_result.x]

    if prev is not None:
        prev_chi = prev.theta * prev.psi
        theta0 = max(prev.theta + eps_theta, theta0)
        p0 = max(p0, 1e-6)
        chi0 = theta0 * p0
        if chi0 < prev_chi + eps_chi:
            p0 = (prev_chi + eps_chi) / theta0

    rho0 = float(np.clip(rho0, -0.99, 0.99))
    p0 = float(np.clip(p0, 1e-6, 20.0))

    u0 = float(np.arctanh(rho0))
    v0 = float(np.log(p0))
    return np.array([theta0, u0, v0], dtype=np.float64)


def _fit_slice(
    points: list[tuple[float, float]],
    prev: SSVIParams | None = None,
    *,
    eps_theta: float = _EPS_THETA,
    eps_chi: float = _EPS_CHI,
) -> SSVIParams:
    """Fit a single SSVI slice subject to hard no-arbitrage constraints.

    Parameters
    ----------
    points : list of (k, w)
        Log-moneyness / total-variance pairs for this slice.
    prev : SSVIParams or None
        Previous (shorter-maturity) slice parameters.  When given the
        Hendriks-Martini calendar constraints are added.
    eps_theta, eps_chi : float
        Small positive floors for the theta and chi monotonicity
        constraints.

    Returns
    -------
    SSVIParams

    Raises
    ------
    RuntimeError
        If the optimizer cannot satisfy all hard constraints.

    Reference: Hendriks & Martini (2019) Prop 3.1; Corbetta et al.
    (2019) Sec 2.2-2.3.
    """
    if len(points) < 5:
        raise ValueError("Need at least 5 points to fit SSVI slice")

    ks = np.array([k for k, _ in points], dtype=np.float64)
    ws = np.array([w for _, w in points], dtype=np.float64)

    x0 = _initial_guess(points, prev, eps_theta, eps_chi)

    # ── Variable bounds ────────────────────────────────────────────
    bounds = Bounds(
        lb=[1e-6, -6.0, float(np.log(1e-8))],
        ub=[10.0,  6.0, float(np.log(20.0))],
    )

    # ── Objective: sum of squared residuals ────────────────────────
    def _objective(x: NDArray[np.float64]) -> float:
        theta, u, v = x
        rho = float(np.tanh(u))
        p = float(np.exp(v))
        return float(np.sum(
            (np.array([ssvi_w(float(k), theta, rho, p) for k in ks]) - ws) ** 2
        ))

    # ── Hard constraints ───────────────────────────────────────────
    constraints = _hard_constraints(prev, eps_theta, eps_chi)

    # ── Optimise ───────────────────────────────────────────────────
    result = _constrained_minimize(_objective, x0, bounds, constraints, minimize)

    theta, u, v = result.x
    return SSVIParams(
        theta=float(theta),
        rho=float(np.tanh(u)),
        psi=float(np.exp(v)),
    )


def _hard_fit_is_degenerate_corner(
    prev: SSVIParams | None,
    params: SSVIParams,
    points: list[tuple[float, float]],
) -> bool:
    """Detect a hard eSSVI fit pinned on the H&M Prop 3.1 boundary.

    A fit is flagged through either of two paths:

    **Path A — baseline available: two signals must AGREE.**

    1. **Boundary proximity** — the hard fit landed within a small
       margin of the H&M calendar-arb boundary that ``_fit_slice``
       enforces with its eps floors:

       - ``theta_delta <= _HM_BOUNDARY_MARGIN_THETA`` (10x eps_theta)
       - ``chi_delta   <= _HM_BOUNDARY_MARGIN_CHI`` (10x eps_chi)
       - ``ratio >= 1 - _HM_BOUNDARY_MARGIN_RATIO``

       where ``theta_delta``/``chi_delta`` are the deltas vs ``prev`` and
       ``ratio`` is ``|rho*chi - rho_prev*chi_prev| / chi_delta``.

    2. **Bad per-slice RMSE** — the hard fit's RMSE over ``points``
       exceeds ``max(_HM_RMSE_RATIO_MAX * unconstrained_rmse,
       _HM_RMSE_FLOOR)``.  The unconstrained per-slice fit
       (:func:`fit_ssvi_slice`) is the RMSE baseline: a
       boundary-adjacent fit that genuinely matches the data (e.g. a
       legitimate near-flat chi pair) has a small RMSE and is NOT
       flagged.

    **Path B — baseline unavailable: boundary proximity alone flags.**
    When the unconstrained baseline fit raises, the RMSE comparison
    cannot be computed — but a fit pinned within the boundary window is
    exactly the knife-edge pattern this check exists to catch, so it IS
    flagged and routed to the fallback rather than silently certified.
    Fits outside the boundary window are never flagged, baseline or no
    baseline.

    A fit flagged here is not a genuine arb-free solution — it is an
    optimizer knife-edge that converged to a feasible-but-wrong corner.
    This is the m66 / mutmut_66 pattern (docs/code_review_findings.md
    §6.7): measured corner theta_delta = 9.99e-10, chi_delta = 1.0000e-6,
    ratio = 0.9998, hard RMSE = 0.0499 vs unconstrained RMSE = 1.6e-11.
    ``fit_ssvi_surface_sequential`` routes flagged fits to the
    unconstrained fallback.

    The first slice has no H&M predecessor boundary to sit on, so
    ``prev=None`` is never flagged.

    Returns
    -------
    bool
        ``True`` iff the fit is within the boundary margin AND (its RMSE
        is anomalously bad relative to the unconstrained fit, OR the
        unconstrained baseline fit is unavailable).
    """
    if prev is None:
        return False

    theta_delta, chi_delta, ratio = _hm_boundary_deltas(prev, params)

    if not _within_boundary_window(theta_delta, chi_delta, ratio):
        # Fast path: far from the boundary, no need for the unconstrained
        # baseline fit.
        return False

    hard_rmse = _slice_rmse(params, points)
    try:
        unconstrained = fit_ssvi_slice(points)
    except (RuntimeError, ValueError):
        # No baseline to compare against — but a fit pinned on the H&M
        # boundary is precisely the m66 pattern that must never be
        # silently certified, so flag it and let the caller route it to
        # the honest fallback.
        return True
    unconstrained_rmse = _slice_rmse(unconstrained, points)
    bad_rmse = hard_rmse > max(
        _HM_RMSE_RATIO_MAX * unconstrained_rmse, _HM_RMSE_FLOOR
    )
    return bad_rmse


def _fit_one_slice(
    expiry: float,
    pts: list[tuple[float, float]],
    last_valid_prev: SSVIParams | None,
) -> tuple[str, SSVIParams | None]:
    """Fit one slice with hard constraints, falling back on failure.

    Returns ``(kind, params)`` where ``kind`` is ``"hard"`` (a
    hard-constrained H&M arb-free fit), ``"fallback"`` (the
    unconstrained per-slice fit), or ``"failed"`` (both fits failed,
    ``params`` is ``None``).

    A hard fit pinned on the H&M boundary corner (m66 pattern) is
    raised as a ``RuntimeError`` so the generic "hard-constrained fit
    failed" warning below still fires, matching the prior inline
    behavior's two-warning degenerate-corner path.
    """
    try:
        params = _fit_slice(pts, prev=last_valid_prev)
        if last_valid_prev is not None and _hard_fit_is_degenerate_corner(last_valid_prev, params, pts):
            _theta_delta, _chi_delta, _ratio = _hm_boundary_deltas(last_valid_prev, params)
            _logger.warning(
                "eSSVI hard fit for T=%.4f is a degenerate H&M boundary corner (theta_delta=%.3e, chi_delta=%.3e, ratio=%.6f, hard_rmse=%.4e); routing to fallback",
                expiry, _theta_delta, _chi_delta, _ratio, _slice_rmse(params, pts),
            )
            raise RuntimeError("hard fit is a degenerate H&M boundary corner")
        return ("hard", params)
    except RuntimeError as e:
        _logger.warning(
            "eSSVI hard-constrained fit failed for T=%.4f (%s); "
            "falling back to unconstrained per-slice fit",
            expiry, e,
        )
    try:
        params = fit_ssvi_slice(pts)
    except (RuntimeError, ValueError) as e2:
        _logger.error(
            "eSSVI fallback fit also failed for T=%.4f (%s); "
            "skipping this slice",
            expiry, e2,
        )
        return ("failed", None)
    return ("fallback", params)


def fit_ssvi_surface_sequential(
    slices_data: list[tuple[float, list[tuple[float, float]]]],
) -> SequentialFitResult:
    """Fit a sequence of SSVI slices with calendar-arb-free constraints.

    Slices are sorted by ascending maturity and fitted one at a time.
    Each slice inherits the Hendriks-Martini Prop 3.1 constraints from
    its predecessor.  Per-slice rho is fully free (tanh-reparametrised,
    no cross-slice functional form).

    When the hard-constrained fit fails for a slice (the optimizer
    cannot satisfy the H&M Prop 3.1 constraints given the data), the
    function falls back to the unconstrained per-slice
    :func:`fit_ssvi_slice`.  The fallback slice is NOT arb-free by
    construction, but the engine reports this honestly via
    ``verify_hm_condition`` and the ``repair_infeasible`` flag.  The
    fallback result is NOT used as ``prev`` for the next slice's
    constraints — ``prev`` remains the last *hard-constrained*
    successful fit.

    A hard-constrained fit that the optimizer reports as successful is
    additionally checked by :func:`_hard_fit_is_degenerate_corner`: a
    fit pinned within eps of the H&M Prop 3.1 boundary whose per-slice
    RMSE is anomalously bad relative to the unconstrained fit (the m66
    degenerate-corner pattern, docs/code_review_findings.md §6.7) is
    routed through the same ``RuntimeError`` fallback path.  When the
    unconstrained baseline fit is unavailable, the boundary proximity
    alone is enough to flag the corner — either way, a feasible-but-wrong
    boundary corner is never silently certified arb-free.

    Parameters
    ----------
    slices_data : list of (expiry_time, points)
        Each element is ``(T, [(k, w), ...])``.  Slices are sorted
        internally by ascending ``T``.

    Returns
    -------
    SequentialFitResult
        Contains ``fitted_slices`` (successful fits, hard-constrained
        or fallback), ``fallback_slices`` (T values where only the
        unconstrained fallback succeeded), and ``failed_slices``
        (T values where both fits failed).

    Reference
    ----------
    Hendriks & Martini (2019) Prop 3.1; Corbetta et al. (2019) Sec 2.2-2.3.
    """
    ordered = sorted(slices_data, key=lambda sd: sd[0])

    fitted: list[tuple[float, SSVIParams]] = []
    fallback: list[float] = []
    failed: list[float] = []
    fitted_slices_prev: list[float | None] = []
    last_valid_prev: SSVIParams | None = None
    last_valid_prev_T: float | None = None

    for expiry, pts in ordered:
        # Record the prev_T that will be used for this slice
        prev_T_for_this_slice = last_valid_prev_T

        kind, params = _fit_one_slice(expiry, pts, last_valid_prev)

        if kind == "hard":
            # update only on hard-constrained success
            last_valid_prev = params
            last_valid_prev_T = expiry  # update for NEXT slice
            fitted.append((expiry, params))
            fitted_slices_prev.append(prev_T_for_this_slice)
        elif kind == "fallback":
            fitted.append((expiry, params))
            fallback.append(expiry)
            fitted_slices_prev.append(prev_T_for_this_slice)
            # do NOT update last_valid_prev / last_valid_prev_T —
            # fallback slices aren't arb-free
        else:  # failed
            failed.append(expiry)
            # failed slices are NOT in fitted_slices, so no entry
            # in fitted_slices_prev

    if len(fitted) >= 2:
        params_only = [p for _, p in fitted]
        if not verify_hm_condition(params_only):
            _logger.warning(
                "verify_hm_condition reports violation after fit; "
                "likely one or more slices fell back to unconstrained. "
                "These are reported as remaining violations."
            )

    return SequentialFitResult(
        fitted_slices=fitted,
        fallback_slices=fallback,
        failed_slices=failed,
        fitted_slices_prev=fitted_slices_prev,
    )
