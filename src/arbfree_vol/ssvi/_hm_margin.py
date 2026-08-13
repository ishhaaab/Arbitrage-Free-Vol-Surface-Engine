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
# feasible-but-wrong corner pinned exactly ON the H&M Prop 3.1 boundary
# (theta_delta = eps_theta, chi_delta = eps_chi, ratio ~ 1.0) with an
# anomalously bad per-slice RMSE, and the optimizer reports it as a
# successful certified arb-free fit.  Measured m66 corner (mutmut_66 /
# the dip fixture): theta_delta = 9.99e-10, chi_delta = 1.0000e-6,
# ratio = 0.9998, hard RMSE = 0.0499 vs unconstrained RMSE = 1.6e-11.
# These thresholds are provisional and pending review — see
# docs/code_review_findings.md §6.7 "Proposed fix direction".
_HM_BOUNDARY_MARGIN_THETA: float = 1e-8   # 10x eps_theta
_HM_BOUNDARY_MARGIN_CHI: float = 1e-5     # 10x eps_chi
_HM_BOUNDARY_MARGIN_RATIO: float = 1e-3
_HM_RMSE_RATIO_MAX: float = 5.0
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


def _within_boundary_window(theta_delta: float, chi_delta: float,
                            ratio: float) -> bool:
    """True iff the hard fit sits within the H&M boundary margin window."""
    return (
        theta_delta <= _HM_BOUNDARY_MARGIN_THETA
        and chi_delta <= _HM_BOUNDARY_MARGIN_CHI
        and ratio >= 1.0 - _HM_BOUNDARY_MARGIN_RATIO
    )
