"""Hendriks & Martini Prop 3.1 boundary-margin primitives (post-fit checks)."""

from math import sqrt
from statistics import mean
from arbfree_vol.ssvi.model import SSVIParams, ssvi_w

# Floors applied to the two Hendriks-Martini calendar constraints in
# ``_fit_slice`` (non-decreasing theta, non-decreasing chi).  Hoisted to
# named module constants so the post-fit H&M margin check below can
# reference the same eps values it must sit 10x away from.
_EPS_THETA: float = 1e-9
_EPS_CHI: float = 1e-6

# ── Post-fit margin check for degenerate H&M boundary corners (m66) ──
# docs/code_review_findings.md §6.7: a hard eSSVI fit can converge to a
# feasible-but-wrong corner pinned ON the H&M Prop 3.1 boundary (theta
# and/or chi pinned at their eps floors, parameters equal to the
# predecessor's) with an anomalously bad per-slice RMSE, and the optimizer
# reports it as a successful certified arb-free fit.
#
# Measured corners:
#
#   Synthetic dip fixture (_DIP_TRUTH_ENGINE, 2026-08-17):
#   | T    | theta_delta | chi_delta | ratio | hard_rmse | unc_rmse  | routing |
#   | 0.5  | 1.000e-09   | 1.000e-06 | 1.0   | 5.05e-02  | 6.27e-09  | fallback (flagged) |
#   | 1.0  | 3.981e-02   | 1.678e-02 | 1.0   | 1.66e-04  | 1.18e-10  | hard, NOT flagged |
#   | 2.0  | 1.000e-09   | 1.000e-06 | 1.0   | 5.00e-02  | ~1.6e-11  | fallback (flagged) |
#
#   Live SPY (2026-08-22, T=0.0932 — ESCAPED the old gate):
#   theta_delta = 5.0e-08, chi_delta = 1.000e-06, ratio = 0.999999,
#   hard params == prev params to 4 decimals, hard_rmse = 6.30e-03 vs
#   unc_rmse = 1.27e-03 (ratio 4.97x).  Certified hard despite being a
#   pure predecessor copy.
#
# Design (revised 2026-08-22 after the live escape):
#   - The window is an OR over the two floor-pinning conditions, NOT an
#     AND over three: a corner only needs ONE floor pinned, and ratio is
#     not discriminative (a copy with rho ~= prev's has ratio ~ 0; one
#     with slightly different rho has ratio ~ 1).
#   - Margins are 100x eps (was 10x): live solver tolerance lands corners
#     at theta_delta ~ 5e-8, not at exactly eps_theta.
#   - _HM_RMSE_RATIO_MAX lowered 5.0 -> 3.0: the measured live corner sits
#     at 4.97x, just under the old wire.  A hard fit 3x+ worse than the
#     unconstrained solution while pinned on a floor is the feasible-but-
#     wrong class; legitimately-constrained slices that poor belong in the
#     honest fallback anyway.
_HM_BOUNDARY_MARGIN_THETA: float = 1e-7   # 100x eps_theta
_HM_BOUNDARY_MARGIN_CHI: float = 1e-4     # 100x eps_chi
_HM_RMSE_RATIO_MAX: float = 3.0
_HM_RMSE_FLOOR: float = 1e-9


def _slice_rmse(
    params: SSVIParams, points: list[tuple[float, float]],
) -> float:
    """Root-mean-square total-variance error of ``params`` over ``points``.

    ``sqrt(mean((ssvi_w(k, theta, rho, psi) - w)^2))`` over every
    ``(k, w)`` in ``points``.  Used by the post-fit H&M margin check to
    compare the hard-constrained fit's per-slice residual with the
    unconstrained fit's baseline.
    """
    return sqrt(mean(
        (ssvi_w(k, params.theta, params.rho, params.psi) - w) ** 2
        for k, w in points
    ))


def _hm_boundary_deltas(
    prev: SSVIParams, params: SSVIParams,
) -> tuple[float, float, float]:
    """Compute the H&M Prop 3.1 boundary deltas for a candidate hard fit.

    Returns ``(theta_delta, chi_delta, ratio)`` where
    ``theta_delta = theta - theta_prev``,
    ``chi_delta = chi - chi_prev`` (``chi = theta * psi``) and
    ``ratio = |rho*chi - rho_prev*chi_prev| / max(chi_delta, eps_chi)``.

    Both the degenerate-corner predicate ``_hard_fit_is_degenerate_corner``
    in ``arbfree_vol.ssvi.term_structure`` and its logging call site in
    ``fit_ssvi_surface_sequential`` must use this single source of truth,
    so the decision and the log can never disagree.
    """
    theta_delta = params.theta - prev.theta
    chi_delta = params.theta * params.psi - prev.theta * prev.psi
    ratio = abs(
        params.rho * params.theta * params.psi
        - prev.rho * prev.theta * prev.psi
    ) / max(chi_delta, _EPS_CHI)
    return theta_delta, chi_delta, ratio


def _within_boundary_window(theta_delta: float, chi_delta: float) -> bool:
    """True iff the hard fit pins EITHER H&M floor within its margin.

    OR semantics, not AND: a degenerate corner only needs one floor
    pinned (theta_delta <= margin OR chi_delta <= margin).  The ratio is
    deliberately absent -- a predecessor copy with rho ~= prev's has
    ratio ~ 0 while one with slightly different rho has ratio ~ 1, so it
    discriminates nothing (see the measured-corners table above).
    """
    return (
        theta_delta <= _HM_BOUNDARY_MARGIN_THETA
        or chi_delta <= _HM_BOUNDARY_MARGIN_CHI
    )
