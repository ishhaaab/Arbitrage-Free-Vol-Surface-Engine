"""Tests for verify_hm_condition_breakdown, fitted_slices_prev, and the
production Gatheral-Jacquier butterfly constraint (strict condition 1).

These tests exercise the per-slice H&M Prop 3.1 sub-condition breakdown
function, the new fitted_slices_prev tracking in SequentialFitResult,
and the strict/non-strict split of the two GJ Theorem 4.2 butterfly
conditions in ``_butterfly_constraints``.
"""

import numpy as np
import pytest

from arbfree_vol.ssvi.model import SSVIParams, gatheral_jacquier_condition
from arbfree_vol.ssvi.term_structure import (
    _butterfly_constraints,
    _GJ_CONDITION1_STRICT_EPS,
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
