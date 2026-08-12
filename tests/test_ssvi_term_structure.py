"""Tests for verify_hm_condition_breakdown, fitted_slices_prev, and the
production Gatheral-Jacquier butterfly constraint (strict condition 1).

These tests exercise the per-slice H&M Prop 3.1 sub-condition breakdown
function, the new fitted_slices_prev tracking in SequentialFitResult,
and the strict/non-strict split of the two GJ Theorem 4.2 butterfly
conditions in ``_butterfly_constraints``.
"""

import numpy as np
import pytest

from arbfree_vol.ssvi.model import SSVIParams, gatheral_jacquier_condition, ssvi_w
from arbfree_vol.ssvi.term_structure import (
    _butterfly_constraints,
    _GJ_CONDITION1_STRICT_EPS,
    _hard_fit_is_degenerate_corner,
    _slice_rmse,
    verify_hm_condition_breakdown,
    SequentialFitResult,
)


# ── Helper to build synthetic fitted_slices ──────────────────────────

def _make_fitted_slices(params_list: list[tuple[float, dict]]) -> list[tuple[float, SSVIParams]]:
    """Build a list of (T, SSVIParams) from (T, dict) pairs."""
    return [(T, SSVIParams(**d)) for T, d in params_list]


# ── Test 1: all conditions pass ─────────────────────────────────────

def test_breakdown_all_pass() -> None:
    """Monotonic sequence (theta, chi increasing, ratio OK) → all conditions pass."""
    fitted = _make_fitted_slices([
        (0.25, dict(theta=0.04, rho=-0.3, psi=0.5)),   # chi=0.02
        (0.50, dict(theta=0.08, rho=-0.2, psi=0.6)),   # chi=0.048
        (1.00, dict(theta=0.14, rho=-0.1, psi=0.65)),  # chi=0.091
    ])

    breakdown = verify_hm_condition_breakdown(fitted)

    assert len(breakdown) == 2  # first slice has no prev
    for entry in breakdown:
        assert entry["theta_ok"] is True
        assert entry["chi_ok"] is True
        assert entry["ratio_ok"] is True
        assert entry["failing_conditions"] == []


# ── Test 2: theta violation ─────────────────────────────────────────

def test_breakdown_theta_violation() -> None:
    """Theta dips at slice 2 → failing_conditions=["theta"]."""
    fitted = _make_fitted_slices([
        (0.25, dict(theta=0.08, rho=-0.3, psi=0.5)),   # chi=0.04
        (0.50, dict(theta=0.04, rho=-0.2, psi=0.6)),   # chi=0.024 — theta dipped
        (1.00, dict(theta=0.14, rho=-0.1, psi=0.65)),  # chi=0.091
    ])

    breakdown = verify_hm_condition_breakdown(fitted)

    # Slice at T=0.50: theta dipped from 0.08 to 0.04
    entry_050 = [e for e in breakdown if e["slice_T"] == 0.50][0]
    assert "theta" in entry_050["failing_conditions"]
    assert entry_050["theta_ok"] is False

    # Slice at T=1.00: theta increased from 0.04 to 0.14 — should pass
    entry_100 = [e for e in breakdown if e["slice_T"] == 1.00][0]
    assert "theta" not in entry_100["failing_conditions"]


# ── Test 3: chi violation ───────────────────────────────────────────

def test_breakdown_chi_violation() -> None:
    """Chi dips at slice 2 (theta OK, chi not) → failing_conditions=["chi"]."""
    # theta increases, but psi drops enough that chi = theta*psi decreases
    fitted = _make_fitted_slices([
        (0.25, dict(theta=0.04, rho=-0.3, psi=0.8)),   # chi=0.032
        (0.50, dict(theta=0.05, rho=-0.2, psi=0.5)),   # chi=0.025 — chi dipped
        (1.00, dict(theta=0.14, rho=-0.1, psi=0.65)),  # chi=0.091
    ])

    breakdown = verify_hm_condition_breakdown(fitted)

    entry_050 = [e for e in breakdown if e["slice_T"] == 0.50][0]
    assert "chi" in entry_050["failing_conditions"]
    assert entry_050["chi_ok"] is False
    assert entry_050["theta_ok"] is True  # theta increased


# ── Test 3b: ratio undefined when chi is non-monotonic ───────────────

def test_breakdown_ratio_undefined_when_chi_dips() -> None:
    """When chi decreases, the ratio condition is undefined: ratio_value
    and ratio_ok are None and "ratio" is NOT a failing condition.  A chi
    dip is the PRIMARY failure; the old clamped-to-tol denominator
    manufactured a huge, misleading ratio that made every chi dip look
    like a ratio violation."""
    # chi: 0.04 -> 0.024 (dips at slice 2)
    fitted = _make_fitted_slices([
        (0.25, dict(theta=0.08, rho=-0.3, psi=0.5)),   # chi=0.04
        (0.50, dict(theta=0.04, rho=-0.2, psi=0.6)),   # chi=0.024 — chi dipped
    ])

    breakdown = verify_hm_condition_breakdown(fitted)

    assert len(breakdown) == 1
    entry = breakdown[0]
    assert entry["chi_ok"] is False
    assert entry["ratio_value"] is None
    assert entry["ratio_ok"] is None
    assert "ratio" not in entry["failing_conditions"]
    assert "chi" in entry["failing_conditions"]


def test_breakdown_ratio_undefined_when_chi_flat() -> None:
    """A flat chi (chi_delta below the tolerance) also leaves the ratio
    undefined — the denominator is at the noise floor, so no ratio
    conclusion is drawn."""
    fitted = _make_fitted_slices([
        (0.25, dict(theta=0.08, rho=-0.3, psi=0.5)),   # chi=0.04
        (0.50, dict(theta=0.08, rho=-0.2, psi=0.5)),   # chi=0.04 — flat
    ])

    breakdown = verify_hm_condition_breakdown(fitted)

    entry = breakdown[0]
    assert entry["theta_ok"] is True
    assert entry["chi_ok"] is True
    assert entry["ratio_value"] is None
    assert entry["ratio_ok"] is None
    assert "ratio" not in entry["failing_conditions"]
    assert entry["failing_conditions"] == []


# ── Test 4: ratio violation ─────────────────────────────────────────

def test_breakdown_ratio_violation() -> None:
    """Theta and chi monotonic, but ratio > 1 → failing_conditions=["ratio"]."""
    # Construct params where rho*chi changes too much relative to chi change
    # rho_prev * chi_prev = -0.9 * 0.02 = -0.018
    # rho_self * chi_self = 0.9 * 0.03 = 0.027
    # |0.027 - (-0.018)| / (0.03 - 0.02) = 0.045 / 0.01 = 4.5 > 1
    fitted = _make_fitted_slices([
        (0.25, dict(theta=0.04, rho=-0.9, psi=0.5)),   # chi=0.02, rho*chi=-0.018
        (0.50, dict(theta=0.06, rho=0.9, psi=0.5)),    # chi=0.03, rho*chi=0.027
    ])

    breakdown = verify_hm_condition_breakdown(fitted)

    assert len(breakdown) == 1
    entry = breakdown[0]
    assert "ratio" in entry["failing_conditions"]
    assert entry["ratio_ok"] is False
    assert entry["theta_ok"] is True
    assert entry["chi_ok"] is True


# ── Test 5: uses actual prev, not adjacent ───────────────────────────

def test_breakdown_uses_actual_prev_not_adjacent() -> None:
    """With consecutive fallbacks, breakdown should compare against the
    actual prev from calibration (fitted_prev_Ts), not the adjacent slice.

    Sequence: [hard1, hard2, fallback3, fallback4, hard5]
    fallback4's prev is hard2 (not fallback3), because the fitter doesn't
    update last_valid_prev on fallback.
    """
    T1, T2, T3, T4, T5 = 0.25, 0.50, 0.75, 1.00, 1.50

    # hard2: theta=0.08, chi=0.048
    # fallback3: theta=0.09, chi=0.054 (passes against hard2)
    # fallback4: theta=0.085, chi=0.051 — chi would PASS against fallback3
    #   (0.051 >= 0.054? NO, 0.051 < 0.054 → chi FAILS against fallback3)
    #   but against hard2: 0.051 >= 0.048 → chi PASSES
    # Let's construct so that fallback4 passes against hard2 but fails against fallback3

    fitted = _make_fitted_slices([
        (T1, dict(theta=0.04, rho=-0.3, psi=0.5)),    # chi=0.02
        (T2, dict(theta=0.08, rho=-0.2, psi=0.6)),    # chi=0.048
        (T3, dict(theta=0.09, rho=-0.15, psi=0.7)),   # chi=0.063
        (T4, dict(theta=0.085, rho=-0.1, psi=0.65)),  # chi=0.05525 — < 0.063
        (T5, dict(theta=0.14, rho=-0.05, psi=0.7)),   # chi=0.098
    ])

    # actual prev_Ts: hard1 has no prev, hard2→hard1, fallback3→hard2,
    # fallback4→hard2 (not fallback3!), hard5→hard2
    fitted_prev_Ts = [None, T1, T2, T2, T2]

    breakdown = verify_hm_condition_breakdown(fitted, fitted_prev_Ts=fitted_prev_Ts)

    # fallback4 (T4) should compare against T2 (hard2), not T3
    entry_T4 = [e for e in breakdown if e["slice_T"] == T4][0]
    assert entry_T4["prev_T"] == T2
    # chi_self=0.05525, chi_prev(T2)=0.048 → chi passes
    assert entry_T4["chi_ok"] is True

    # If we had used adjacent (T3), chi_self=0.05525 < chi_prev=0.063 → would fail
    # So the test confirms we're using the ACTUAL prev, not adjacent


# ── Test 6: legacy adjacent pairs when fitted_prev_Ts=None ──────────

def test_breakdown_legacy_adjacent_pairs() -> None:
    """When fitted_prev_Ts=None, function falls back to adjacent pairs."""
    fitted = _make_fitted_slices([
        (0.25, dict(theta=0.04, rho=-0.3, psi=0.5)),
        (0.50, dict(theta=0.08, rho=-0.2, psi=0.6)),
        (1.00, dict(theta=0.14, rho=-0.1, psi=0.65)),
    ])

    breakdown = verify_hm_condition_breakdown(fitted, fitted_prev_Ts=None)

    assert len(breakdown) == 2
    assert breakdown[0]["prev_T"] == 0.25
    assert breakdown[1]["prev_T"] == 0.50


# ── Test 7: single slice → empty list ───────────────────────────────

def test_breakdown_no_prev() -> None:
    """Single slice → empty list (no predecessor)."""
    fitted = _make_fitted_slices([
        (0.25, dict(theta=0.04, rho=-0.3, psi=0.5)),
    ])

    breakdown = verify_hm_condition_breakdown(fitted)
    assert breakdown == []


# ── Test 8: prev not in fitted → that slice is skipped ──────────────

def test_breakdown_prev_not_in_fitted() -> None:
    """When fitted_prev_Ts references a T not in fitted_slices, that slice is skipped."""
    fitted = _make_fitted_slices([
        (0.25, dict(theta=0.04, rho=-0.3, psi=0.5)),
        (0.50, dict(theta=0.08, rho=-0.2, psi=0.6)),
    ])

    # prev_T=0.999 is not in fitted_slices
    breakdown = verify_hm_condition_breakdown(fitted, fitted_prev_Ts=[None, 0.999])

    # Only the first slice has no prev (skipped), the second has prev=0.999
    # which is not in params_by_T → also skipped
    assert len(breakdown) == 0


# ── Test 9: fitted_slices_prev tracking ──────────────────────────────

def test_fitted_slices_prev_field_exists() -> None:
    """SequentialFitResult has fitted_slices_prev field with default empty list."""
    result = SequentialFitResult(
        fitted_slices=[],
        fallback_slices=[],
        failed_slices=[],
    )
    assert hasattr(result, "fitted_slices_prev")
    assert result.fitted_slices_prev == []


def test_fitted_slices_prev_populated() -> None:
    """SequentialFitResult accepts fitted_slices_prev in constructor."""
    p1 = SSVIParams(theta=0.04, rho=-0.3, psi=0.5)
    p2 = SSVIParams(theta=0.08, rho=-0.2, psi=0.6)
    result = SequentialFitResult(
        fitted_slices=[(0.25, p1), (0.50, p2)],
        fallback_slices=[],
        failed_slices=[],
        fitted_slices_prev=[None, 0.25],
    )
    assert result.fitted_slices_prev == [None, 0.25]


# ── Gatheral-Jacquier strict condition-1 production constraint ────────

def test_butterfly_constraints_condition1_strict_boundary() -> None:
    """A slice at EXACT condition-1 equality must be rejected by the
    production constraint: GJ Theorem 4.2 condition 1
    (theta*psi*(1+|rho|) < 4) is STRICT.  The condition-1 residual is
    shifted by the strict eps so it comes out negative at the boundary,
    while the condition-2 residual stays at exactly 4 - lhs (non-strict).

    rho=0.5, p=1.0, theta=4/(1+rho)/p makes theta*p*(1+rho) == 4.0
    exactly, so the (1+rho) branch of condition 1 binds."""
    rho, p = 0.5, 1.0
    theta = 4.0 / ((1.0 + rho) * p)
    res = _butterfly_constraints(theta, rho, p)

    # Condition 1, (1+rho) branch: unshifted value is exactly 0, so the
    # strict eps shift makes the residual exactly -eps (< 0, rejected).
    unshifted_c1 = 4.0 - theta * p * (1.0 + rho)
    assert unshifted_c1 == 0.0
    assert res[0] == pytest.approx(
        unshifted_c1 - _GJ_CONDITION1_STRICT_EPS, abs=1e-15
    )
    assert res[0] == pytest.approx(-_GJ_CONDITION1_STRICT_EPS, abs=1e-15)
    assert res[0] < 0.0

    # Condition 2, (1+rho) branch: theta*p^2*(1+rho) == 4.0 exactly too,
    # but condition 2 is non-strict so the residual is exactly 0 (>= 0,
    # accepted) — no eps shift.
    unshifted_c2 = 4.0 - theta * p * p * (1.0 + rho)
    assert unshifted_c2 == 0.0
    assert res[2] == pytest.approx(unshifted_c2, abs=1e-15)
    assert res[2] >= 0.0

    # The residual delta vs the unshifted values is exactly the eps on
    # both condition-1 residuals and exactly 0 on both condition-2 ones.
    assert res[1] == pytest.approx(
        4.0 - theta * p * (1.0 - rho) - _GJ_CONDITION1_STRICT_EPS, abs=1e-15
    )
    assert res[3] == pytest.approx(4.0 - theta * p * p * (1.0 - rho), abs=1e-15)
    assert res[1] > 0.0
    assert res[3] > 0.0


def test_butterfly_constraints_condition2_nonstrict_boundary() -> None:
    """A slice at EXACT condition-2 equality must be accepted (non-strict
    in GJ Theorem 4.2), while condition 1 still holds with a positive
    margin — this is the "condition 2 may touch the boundary, condition 1
    may not" distinction.

    rho=0.5, p=2.0, theta=4/((1+rho)*p^2) makes theta*p^2*(1+rho) == 4.0
    exactly; condition 1 is then theta*p*(1+rho) == 2.0 < 4.0."""
    rho, p = 0.5, 2.0
    theta = 4.0 / ((1.0 + rho) * p * p)
    res = _butterfly_constraints(theta, rho, p)

    # Condition 2, (1+rho) branch: exactly 0 (>= 0, accepted, unshifted).
    assert res[2] == pytest.approx(0.0, abs=1e-15)

    # Condition 1 still strictly inside: theta*p*(1+rho) == 2.0 < 4.0, so
    # even after the strict eps shift the residual stays positive.
    assert res[0] == pytest.approx(2.0 - _GJ_CONDITION1_STRICT_EPS, abs=1e-12)
    assert res[0] > 0.0

    # The whole slice is feasible: every residual is >= 0.
    assert np.min(res) >= 0.0


def test_butterfly_constraints_strict_mode_matches_gj_diagnostic() -> None:
    """The production residual signs agree with the approved strict-mode
    diagnostic ``gatheral_jacquier_condition(strict=True)``: healthy and
    condition-2-boundary slices are safe in both, and condition-1-boundary
    slices are NOT safe in both (production residual negative)."""
    cases = [
        (0.04, -0.4, 0.5, "healthy"),
        (8.0, 0.0, 0.5, "condition-1 equality, rho=0"),
        (8.0 / 3.0, 0.5, 1.0, "condition-1 equality, (1+rho) branch"),
        (1.0, 0.0, 2.0, "condition-2 equality, rho=0"),
        (2.0 / 3.0, 0.5, 2.0, "condition-2 equality, (1+rho) branch"),
    ]
    for theta, rho, psi, label in cases:
        prod = _butterfly_constraints(theta, rho, psi)
        prod_safe = bool(np.min(prod) >= 0.0)
        diag = gatheral_jacquier_condition(theta, rho, psi, strict=True)
        assert prod_safe == (diag >= 0.0), (
            f"{label}: production safe={prod_safe} but diagnostic "
            f"strict={diag} (safe={diag >= 0.0})"
        )
        if label.startswith("condition-1"):
            # The strict diagnostic must flag the boundary as not safe and
            # the production condition-1 residual must be negative.
            assert diag < 0.0
            assert np.min(prod) < 0.0


# ── Post-fit margin check for degenerate H&M boundary corners (m66) ──────
# docs/code_review_findings.md §6.7: a hard fit pinned exactly on the H&M
# Prop 3.1 boundary (theta_delta ~ eps_theta, chi_delta ~ eps_chi,
# ratio ~ 0.9998) with an anomalously bad RMSE must be routed to the
# fallback path.  These unit tests pin the helper's two-signal logic
# (boundary proximity AND bad RMSE must AGREE).


def _m66_corner_params(prev: SSVIParams) -> SSVIParams:
    """Build params pinned at ``prev`` on the H&M boundary, reproducing
    the m66 corner: theta_delta ~ 1e-9, chi_delta ~ 1e-6, ratio ~ 0.9998."""
    chi_prev = prev.theta * prev.psi
    theta = prev.theta + 1e-9
    chi = chi_prev + 1e-6
    psi = chi / theta
    rho = (prev.rho * chi_prev + 0.9998 * 1e-6) / chi
    return SSVIParams(theta=theta, rho=rho, psi=psi)


def test_hard_fit_is_degenerate_corner_m66_scenario() -> None:
    """The m66 corner must be flagged: a hard fit pinned at the H&M eps
    floors whose per-slice RMSE (~0.05) is orders of magnitude worse than
    the unconstrained fit's (~1e-13), over points whose true w(k) is the
    DIP truth (theta=0.07)."""
    prev = SSVIParams(theta=0.119252, rho=0.08325, psi=0.47565)
    params = _m66_corner_params(prev)

    # Sanity: the constructed params really are the m66 corner.
    theta_delta = params.theta - prev.theta
    chi_delta = params.theta * params.psi - prev.theta * prev.psi
    ratio = abs(
        params.rho * params.theta * params.psi
        - prev.rho * prev.theta * prev.psi
    ) / max(chi_delta, 1e-6)
    assert theta_delta == pytest.approx(1e-9, rel=1e-6)
    assert chi_delta == pytest.approx(1e-6, rel=1e-6)
    assert ratio == pytest.approx(0.9998, abs=1e-4)

    ks = np.linspace(-1.0, 1.0, 9)
    points = [(float(k), ssvi_w(float(k), 0.07, 0.2, 0.55)) for k in ks]

    assert _hard_fit_is_degenerate_corner(prev, params, points) is True


def test_hard_fit_is_degenerate_corner_healthy_not_flagged() -> None:
    """A healthy pair (theta well inside the boundary) with points
    generated from the fit must NOT be flagged."""
    prev = SSVIParams(theta=0.04, rho=-0.4, psi=0.5)
    params = SSVIParams(theta=0.08, rho=-0.4, psi=0.5)
    ks = np.linspace(-1.0, 1.0, 9)
    points = [
        (float(k), ssvi_w(float(k), params.theta, params.rho, params.psi))
        for k in ks
    ]

    assert _hard_fit_is_degenerate_corner(prev, params, points) is False


def test_hard_fit_near_boundary_but_good_rmse_not_flagged() -> None:
    """A fit pinned on the H&M boundary whose RMSE is genuinely small
    (points generated FROM the fit) must NOT be flagged: the RMSE
    secondary signal prevents false positives on legitimate
    boundary-adjacent fits."""
    prev = SSVIParams(theta=0.04, rho=-0.3, psi=0.5)
    params = _m66_corner_params(prev)
    ks = np.linspace(-1.0, 1.0, 9)
    points = [
        (float(k), ssvi_w(float(k), params.theta, params.rho, params.psi))
        for k in ks
    ]

    # Sanity: this fit really is near the H&M boundary.
    theta_delta = params.theta - prev.theta
    chi_delta = params.theta * params.psi - prev.theta * prev.psi
    assert theta_delta <= 1e-8
    assert chi_delta <= 1e-5
    assert _slice_rmse(params, points) == pytest.approx(0.0, abs=1e-12)

    assert _hard_fit_is_degenerate_corner(prev, params, points) is False


def test_hard_fit_is_degenerate_corner_first_slice_not_flagged() -> None:
    """prev=None must not be flagged: a first slice has no H&M
    predecessor boundary to sit on."""
    params = SSVIParams(theta=0.08, rho=-0.3, psi=0.5)
    ks = np.linspace(-1.0, 1.0, 9)
    points = [(float(k), ssvi_w(float(k), 0.07, 0.2, 0.55)) for k in ks]

    assert _hard_fit_is_degenerate_corner(None, params, points) is False


def test_slice_rmse_zero_on_exact_fit() -> None:
    """_slice_rmse returns ~0 when params reproduce the points exactly."""
    params = SSVIParams(theta=0.08, rho=-0.4, psi=0.5)
    ks = np.linspace(-1.0, 1.0, 9)
    points = [
        (float(k), ssvi_w(float(k), params.theta, params.rho, params.psi))
        for k in ks
    ]

    assert _slice_rmse(params, points) == pytest.approx(0.0, abs=1e-12)
