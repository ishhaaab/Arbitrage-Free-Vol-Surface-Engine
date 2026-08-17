"""Tests for the eSSVI sequential term-structure fitter.

Verifies the Hendriks & Martini (2019) Prop 3.1 no-calendar-spread
conditions hold after calibration.

Reference: Hendriks & Martini (2019), "The Extended SSVI Volatility
Surface", J. Comput. Finance 22(5), 25-39, Prop 3.1.
"""

import numpy as np
import pytest

from arbfree_vol.ssvi.model import SSVIParams, ssvi_w, to_raw_svi_params
from arbfree_vol.ssvi.term_structure import (
    _GJ_CONDITION1_STRICT_EPS,
    _butterfly_constraints,
    fit_ssvi_surface_sequential,
    verify_hm_condition,
    verify_hm_condition_breakdown,
    verify_ssvi_calendar_free,
)
from arbfree_vol.arbitrage.svi_detect import detect_svi_surface

from tests.repair_helpers import _DIP_TRUTH_ENGINE


# ── Synthetic ground truth ──────────────────────────────────────────
# Three slices with theta increasing and chi = theta*psi increasing.
_TRUTH = [
    dict(theta=0.04, rho=-0.3, psi=0.5),   # T=0.25
    dict(theta=0.08, rho=-0.2, psi=0.6),   # T=0.50
    dict(theta=0.14, rho=-0.1, psi=0.65),  # T=1.00
]

_EXPIRIES = [0.25, 0.50, 1.00]

_KS = np.linspace(-1.0, 1.0, 9).tolist()


def _make_slices_data():
    """Build synthetic (expiry, [(k, w)]) data from ground-truth SSVI."""
    slices = []
    for T, truth in zip(_EXPIRIES, _TRUTH):
        pts = [
            (float(k), ssvi_w(float(k), truth["theta"], truth["rho"], truth["psi"]))
            for k in _KS
        ]
        slices.append((T, pts))
    return slices


def _sequential_fit_params(slices_data):
    """Run the sequential fitter and return just the fitted params."""
    result = fit_ssvi_surface_sequential(slices_data)
    return [p for _, p in result.fitted_slices]


# ── Test 1 ──────────────────────────────────────────────────────────
def test_theta_and_chi_non_decreasing() -> None:
    """After calibration, theta_i and chi_i=theta_i*psi_i must be
    non-decreasing across slices (Hendriks & Martini 2019, Prop 3.1,
    conditions (a) and (b))."""
    params = _sequential_fit_params(_make_slices_data())

    assert len(params) == 3

    for i in range(len(params) - 1):
        assert params[i + 1].theta >= params[i].theta - 1e-8, (
            f"theta not non-decreasing at slice {i}: "
            f"{params[i].theta} > {params[i + 1].theta}"
        )
        chi_i = params[i].theta * params[i].psi
        chi_next = params[i + 1].theta * params[i + 1].psi
        assert chi_next >= chi_i - 1e-8, (
            f"chi not non-decreasing at slice {i}: "
            f"{chi_i} > {chi_next}"
        )


# ── Test 2 ──────────────────────────────────────────────────────────
def test_pairwise_inequality_holds() -> None:
    """For each adjacent pair the discrete Prop 3.1 inequality must hold:

    | rho_{i+1}*chi_{i+1} - rho_i*chi_i | / (chi_{i+1} - chi_i) <= 1

    (Hendriks & Martini 2019, Prop 3.1, condition (c)).
    """
    params = _sequential_fit_params(_make_slices_data())

    chis = [p.theta * p.psi for p in params]
    tol = 1e-8

    for i in range(len(params) - 1):
        denom = chis[i + 1] - chis[i]
        denom = max(denom, tol)
        num = params[i + 1].rho * chis[i + 1] - params[i].rho * chis[i]
        ratio = abs(num) / denom
        assert ratio <= 1.0 + tol, (
            f"pairwise inequality violated at pair ({i}, {i+1}): "
            f"|ratio| = {ratio:.10f} > 1"
        )


# ── Test 3 ──────────────────────────────────────────────────────────
def test_grid_calendar_detector_reports_zero() -> None:
    """The grid-based detect_svi_surface must report zero violations on
    a calibrated eSSVI surface (redundant regression check)."""
    params = _sequential_fit_params(_make_slices_data())

    svi_pairs: list[tuple[float, object]] = []
    for T, p in zip(_EXPIRIES, params):
        a, b, rho, m, sigma = to_raw_svi_params(p.theta, p.rho, p.psi)
        from arbfree_vol.svi.model import SVIParams
        svi_pairs.append((T, SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)))

    report = detect_svi_surface(svi_pairs)
    assert report.is_arbitrage_free, (
        f"detect_svi_surface found {len(report.violations)} violation(s) "
        f"on a calibrated eSSVI surface: "
        f"{[v.detail for v in report.violations]}"
    )


# ── Test 4 ──────────────────────────────────────────────────────────
def test_near_equal_chi_no_divide_by_zero() -> None:
    """When chi_2 is very close to chi_1 the fitter must not produce
    NaN/inf and verify_hm_condition must still return True.

    We feed a near-flat surface (identical w across two expiries) so
    the optimizer is pushed towards chi_2 ≈ chi_1.  The eps_chi floor
    prevents division by zero in the pairwise inequality constraint.
    """
    # Two slices with very similar data => chi should stay close
    flat_w = 0.04  # constant total variance
    pts = [(float(k), flat_w) for k in np.linspace(-1.0, 1.0, 9)]
    slices_data = [
        (0.25, pts),
        (0.50, pts),  # identical points => forces chi close
    ]

    params = fit_ssvi_surface_sequential(slices_data)
    assert len(params.fitted_slices) == 2

    # All params must be finite
    for _T, p in params.fitted_slices:
        assert np.isfinite(p.theta), f"theta is not finite: {p.theta}"
        assert np.isfinite(p.rho), f"rho is not finite: {p.rho}"
        assert np.isfinite(p.psi), f"psi is not finite: {p.psi}"

    # verify_hm_condition must pass (the eps_chi floor handles denom → 0)
    assert verify_hm_condition([p for _, p in params.fitted_slices]), (
        "verify_hm_condition returned False on near-equal-chi slices"
    )


# ── Test 5: fallback on infeasible slice ────────────────────────────
def test_sequential_fit_falls_back_on_infeasible_slice(monkeypatch) -> None:
    """When a slice's data makes the H&M constraints infeasible against
    the predecessor, the fitter must fall back to the unconstrained
    per-slice fit rather than raising.

    We build a 3-slice surface with normal data and monkeypatch
    ``_fit_slice`` to raise ``RuntimeError`` on the second call (the
    middle slice), simulating an infeasible H&M constraint.  The
    sequential fitter must fall back to ``fit_ssvi_slice``.

    The result must contain all 3 slices.  The fallback slice is NOT
    arb-free by construction, so verify_hm_condition may report a
    violation — that's expected and honest.
    """
    import arbfree_vol.ssvi.term_structure as ts

    slices_data = _make_slices_data()

    # Monkeypatch _fit_slice to fail on the 2nd call
    call_count = {"n": 0}
    _real_fit_slice = ts._fit_slice

    def _failing_fit_slice(points, prev=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated infeasible H&M constraints")
        return _real_fit_slice(points, prev=prev, **kwargs)

    monkeypatch.setattr(ts, "_fit_slice", _failing_fit_slice)

    result = fit_ssvi_surface_sequential(slices_data)

    # All 3 slices must be present (hard or fallback)
    assert len(result.fitted_slices) == 3, (
        f"expected 3 entries, got {len(result.fitted_slices)}"
    )

    # The middle slice (T=0.50) should have fallen back to unconstrained
    assert 0.50 in result.fallback_slices, (
        f"expected T=0.50 in fallback_slices, got {result.fallback_slices}"
    )
    # No slices should have failed entirely
    assert len(result.failed_slices) == 0, (
        f"expected 0 failed slices, got {result.failed_slices}"
    )

    # verify_hm_condition may be False because of the fallback; the point
    # is that it runs without raising on a fallback-containing fit.
    verify_hm_condition([p for _, p in result.fitted_slices])


# ── Test 6: failed slice when both fits fail ─────────────────────────
def test_sequential_fit_reports_failed_slice_when_both_fits_fail(monkeypatch) -> None:
    """When both the hard-constrained fit AND the unconstrained fallback
    fail for a slice, the expiry must appear in ``failed_slices`` and
    NOT in ``fitted_slices``.

    We monkeypatch ``_fit_slice`` to raise on the 2nd call and
    ``fit_ssvi_slice`` to always raise, so the middle slice fails
    entirely.
    """
    import arbfree_vol.ssvi.term_structure as ts

    slices_data = _make_slices_data()

    # Monkeypatch _fit_slice to fail on the 2nd call
    call_count = {"n": 0}
    _real_fit_slice = ts._fit_slice

    def _failing_fit_slice(points, prev=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated infeasible H&M constraints")
        return _real_fit_slice(points, prev=prev, **kwargs)

    # Monkeypatch fit_ssvi_slice to always raise
    def _always_fail_ssvi_slice(points):
        raise RuntimeError("simulated unconstrained fit failure")

    monkeypatch.setattr(ts, "_fit_slice", _failing_fit_slice)
    monkeypatch.setattr(ts, "fit_ssvi_slice", _always_fail_ssvi_slice)

    result = fit_ssvi_surface_sequential(slices_data)

    # Only 2 slices should have fitted (the 1st and 3rd)
    assert len(result.fitted_slices) == 2, (
        f"expected 2 fitted slices, got {len(result.fitted_slices)}"
    )

    # T=0.50 should be in failed_slices
    assert 0.50 in result.failed_slices, (
        f"expected T=0.50 in failed_slices, got {result.failed_slices}"
    )

    # T=0.50 should NOT be in fallback_slices
    assert 0.50 not in result.fallback_slices, (
        f"T=0.50 should not be in fallback_slices, got {result.fallback_slices}"
    )

    # T=0.50 should NOT be in fitted_slices
    fitted_Ts = [T for T, _ in result.fitted_slices]
    assert 0.50 not in fitted_Ts, (
        f"T=0.50 should not be in fitted_slices, got {fitted_Ts}"
    )

    # The other two slices should be present
    assert 0.25 in fitted_Ts
    assert 1.00 in fitted_Ts

    # No fallback slices at all
    assert len(result.fallback_slices) == 0, (
        f"expected 0 fallback slices, got {result.fallback_slices}"
    )


# ── Test 7: SLSQP non-converged statuses must fall back ──────────────
# Regression tests for the SLSQP status-acceptance bug: the SLSQP retry
# used to accept exit modes 0/1/2/3 as success.  Per scipy's
# _slsqp_py.py exit-mode table, only mode 0 ("Optimization terminated
# successfully") is convergence — `result.success` is True exactly for
# mode 0.  Modes 1 (stalled line search), 2 (degenerate problem) and 3
# (LSQ-subproblem iteration cap) are NOT convergence; the fit must
# raise so the caller routes the slice into fallback_slices instead of
# certifying a non-converged fit as arb-free.
def _fit_points():
    """Minimal 5-point (k, w) slice for _fit_slice."""
    return [(float(k), 0.05 + 0.01 * float(k)) for k in (-0.2, -0.1, 0.0, 0.1, 0.2)]


def _scripted_minimize(calls):
    """Build a fake scipy minimize that returns results from an iterator.

    _fit_slice calls minimize twice: trust-constr first, then SLSQP.
    """
    import arbfree_vol.ssvi.term_structure as ts

    it = iter(calls)

    def fake_minimize(fun, x0, method=None, bounds=None, constraints=None, options=None):
        return next(it)

    return ts, fake_minimize


def _opt_result(status, message, success=False):
    from scipy.optimize import OptimizeResult

    return OptimizeResult(
        x=np.array([0.1, 0.0, -0.5]),
        success=success,
        status=status,
        message=message,
    )


def test_fit_slice_slsqp_non_converged_status_raises(monkeypatch) -> None:
    """SLSQP exit mode 1 (stalled line search) must raise RuntimeError,
    not be accepted as a hard-constrained fit."""
    ts, fake_minimize = _scripted_minimize([
        _opt_result(4, "The maximum number of function evaluations is exceeded."),
        _opt_result(1, "Function evaluation required (f & c)"),
    ])
    monkeypatch.setattr(ts, "minimize", fake_minimize)

    with pytest.raises(RuntimeError, match="Function evaluation required"):
        ts._fit_slice(_fit_points())


def test_fit_slice_slsqp_mode8_linesearch_failure_raises(monkeypatch) -> None:
    """SLSQP exit mode 8 ("Positive directional derivative for
    linesearch") is the known RuntimeError surface and must keep
    raising so the caller falls back (regression guard)."""
    ts, fake_minimize = _scripted_minimize([
        _opt_result(4, "The maximum number of function evaluations is exceeded."),
        _opt_result(8, "Positive directional derivative for linesearch"),
    ])
    monkeypatch.setattr(ts, "minimize", fake_minimize)

    with pytest.raises(RuntimeError, match="Positive directional derivative"):
        ts._fit_slice(_fit_points())


def test_fit_slice_trust_constr_success_status_accepted(monkeypatch) -> None:
    """A converged trust-constr fit (success=True, status 1) returns
    fitted parameters without ever reaching the SLSQP retry."""
    from math import exp

    ts, fake_minimize = _scripted_minimize([
        _opt_result(1, "Optimization terminated successfully", success=True),
    ])
    monkeypatch.setattr(ts, "minimize", fake_minimize)

    params = ts._fit_slice(_fit_points())
    assert params.theta == pytest.approx(0.1)
    assert params.rho == pytest.approx(0.0)
    assert params.psi == pytest.approx(exp(-0.5))


def test_fit_slice_slsqp_success_status_zero_accepted(monkeypatch) -> None:
    """A converged SLSQP retry (success=True, exit mode 0) must be
    accepted as a hard-constrained fit.

    The trust-constr attempt fails (status 4, max f-evals); the SLSQP
    retry converges with exit mode 0 ("Optimization terminated
    successfully") — the ONLY SLSQP mode that counts as convergence.
    This genuinely reaches the SLSQP branch, unlike a scripted
    trust-constr-only success."""
    from math import exp

    ts, fake_minimize = _scripted_minimize([
        _opt_result(4, "The maximum number of function evaluations is exceeded."),
        _opt_result(0, "Optimization terminated successfully", success=True),
    ])
    monkeypatch.setattr(ts, "minimize", fake_minimize)

    params = ts._fit_slice(_fit_points())
    assert params.theta == pytest.approx(0.1)
    assert params.rho == pytest.approx(0.0)
    assert params.psi == pytest.approx(exp(-0.5))


def test_slsqp_non_converged_slice_lands_in_fallback(monkeypatch) -> None:
    """End-to-end: when the SLSQP retry returns a non-converged status,
    the slice routes into fallback_slices (same as the RuntimeError
    path) instead of being certified as a hard-constrained fit."""
    ts, fake_minimize = _scripted_minimize([
        _opt_result(1, "Function evaluation required (f & c)"),
        _opt_result(1, "Function evaluation required (f & c)"),
        _opt_result(1, "Function evaluation required (f & c)"),
        _opt_result(1, "Function evaluation required (f & c)"),
    ])
    monkeypatch.setattr(ts, "minimize", fake_minimize)

    slices_data = [
        (0.25, _fit_points()),
        (0.50, _fit_points()),
    ]
    result = fit_ssvi_surface_sequential(slices_data)

    assert result.fallback_slices == [0.25, 0.50], (
        f"expected both slices in fallback_slices, got {result.fallback_slices}"
    )
    assert result.failed_slices == []
    fitted_Ts = [T for T, _ in result.fitted_slices]
    assert fitted_Ts == [0.25, 0.50]


# ── Test 8: hard-constraint enforcement on genuinely incompatible data ─
# End-to-end regression for the sequential hard-constraint logic.  The
# dataset below is genuinely H&M-incompatible (slice 2 dips in theta AND
# chi against slice 1, slice 4 dips against slice 3) — nothing is
# monkeypatched.  If the constraints in _fit_slice were deleted, the
# unconstrained fit would recover the dips and verify_hm_condition
# would report a violation with an EMPTY fallback_slices; the tests
# below would fail.
def _dip_slices_data(with_tail: bool = False):
    """Build (expiry, [(k, w)]) data from the dip ground truth."""
    data = [
        (T, [(float(k), ssvi_w(float(k), t["theta"], t["rho"], t["psi"])) for k in _KS])
        for T, t in _DIP_TRUTH_ENGINE
    ]
    if with_tail:
        data.append(
            (3.0, [(float(k), ssvi_w(float(k), 0.10, 0.1, 0.5)) for k in _KS])
        )
    return data


def test_hard_constraint_or_fallback_on_incompatible_data() -> None:
    """On genuinely H&M-incompatible data, every fitted slice either
    satisfies Prop 3.1 or is honestly recorded in fallback_slices.

    Deleting the hard constraints in _fit_slice makes the unconstrained
    fit recover the theta dips → verify_hm_condition() is False with an
    empty fallback_slices → this test fails.

    Deterministic concrete outcome for ``_DIP_TRUTH_ENGINE`` (verified): the
    theta-dipping slices T=0.5 and T=2.0 land on the H&M boundary corner
    (theta_delta=1e-9, chi_delta=1e-6, ratio≈1) and are routed to the
    unconstrained fallback; T=0.25 and T=1.0 fit hard.  The fallback set
    is asserted exactly, not conditionally.
    """
    result = fit_ssvi_surface_sequential(_dip_slices_data())

    assert result.failed_slices == []
    assert result.fallback_slices == [0.5, 2.0], (
        f"expected the theta-dipping slices to fall back exactly once "
        f"(T=0.5 and T=2.0), got {result.fallback_slices}"
    )
    fitted_Ts = [T for T, _ in result.fitted_slices]
    assert fitted_Ts == [0.25, 0.5, 1.0, 2.0], (
        f"expected all four slices fitted, got {fitted_Ts}"
    )
    params_only = [p for _, p in result.fitted_slices]
    assert not verify_hm_condition(params_only), (
        "the two fallback slices are not arb-free by construction, so "
        "verify_hm_condition must report a violation"
    )


def test_fitted_slices_prev_threads_last_hard_constrained_predecessor() -> None:
    """fitted_slices_prev must point at the last HARD-CONSTRAINED slice
    before each fitted slice, skipping fallback slices.

    Deterministic concrete outcome for ``_DIP_TRUTH_ENGINE`` + the T=3.0 tail
    (verified across repeated runs): T=0.5, T=2.0 and T=3.0 all route to
    the unconstrained fallback via the degenerate H&M boundary-corner
    check, so the predecessor chain skips them:

        prev = [None, 0.25, 0.25, 1.0, 1.0]

    T=1.0 fits hard and anchors both T=2.0 (fallback) and T=3.0
    (fallback) — a regression where a fallback slice wrongly becomes the
    predecessor of the next slice changes this exact sequence.
    """
    result = fit_ssvi_surface_sequential(_dip_slices_data(with_tail=True))

    assert len(result.fitted_slices_prev) == len(result.fitted_slices)
    assert [T for T, _ in result.fitted_slices] == [0.25, 0.5, 1.0, 2.0, 3.0]
    assert result.fallback_slices == [0.5, 2.0, 3.0], (
        f"expected T=0.5, T=2.0 and T=3.0 to fall back, got "
        f"{result.fallback_slices}"
    )
    assert result.fitted_slices_prev == [None, 0.25, 0.25, 1.0, 1.0], (
        "fitted_slices_prev must point at the last HARD-CONSTRAINED "
        "predecessor (skipping fallback slices):\n"
        f"  got      {result.fitted_slices_prev}\n"
        f"  expected [None, 0.25, 0.25, 1.0, 1.0]"
    )


# ── Native calendar gate ─────────────────────────────────────────────
# verify_hm_condition checks necessary (not sufficient) conditions: a
# fitted pair can satisfy theta/chi monotonicity and |ratio| <= 1 yet
# still cross in the wings.  verify_ssvi_calendar_free checks the actual
# no-calendar-spread condition on the native SSVI slices over a dense
# grid, catching what the parameter check (and the [-1.5, 1.5] raw-SVI
# detector grid) misses.
def test_verify_ssvi_calendar_free_rejects_crossing_pairs() -> None:
    """The native calendar gate must reject pairs that cross in the
    wings even though the H&M parameter conditions pass."""
    from arbfree_vol.ssvi.term_structure import verify_ssvi_calendar_free

    crossing_pairs = [
        # Architecture-review counterexample: verify_hm_condition=True
        # but min(w2-w1) = -0.00153 at k=0.68.
        [
            SSVIParams(theta=0.0149505446, rho=-0.6548551, psi=0.11491999),
            SSVIParams(theta=0.0574982989, rho=-0.8830506, psi=2.5226500),
        ],
        # Theta == 1 case: curves cross immediately off ATM.
        [
            SSVIParams(theta=1.0, rho=0.90, psi=1.0),
            SSVIParams(theta=1.0, rho=0.81, psi=1.2),
        ],
        # Wing crossing at k ~ -1.73, outside the [-1.5, 1.5] raw-SVI
        # detector grid used by detect_svi_surface.
        [
            SSVIParams(theta=0.5, rho=0.00, psi=1.0),
            SSVIParams(theta=1.0, rho=0.58, psi=1.2),
        ],
    ]
    for pair in crossing_pairs:
        assert verify_hm_condition(pair), (
            f"setup error: H&M parameter conditions should pass for {pair}"
        )
        assert not verify_ssvi_calendar_free(pair), (
            f"calendar gate must reject crossing pair: {pair}"
        )


def test_verify_ssvi_calendar_free_passes_benign_fit() -> None:
    """The native calendar gate passes on a cleanly fitted surface."""
    from arbfree_vol.ssvi.term_structure import verify_ssvi_calendar_free

    result = fit_ssvi_surface_sequential(_make_slices_data())
    params = [p for _, p in result.fitted_slices]
    assert result.fallback_slices == []
    assert verify_ssvi_calendar_free(params)


def test_verify_ssvi_calendar_free_near_equal_chi() -> None:
    """The gate passes on the near-flat (near-equal-chi) surface and does
    not false-positive on numerical noise around chi_2 ~ chi_1."""
    flat_w = 0.04
    pts = [(float(k), flat_w) for k in np.linspace(-1.0, 1.0, 9)]
    result = fit_ssvi_surface_sequential([(0.25, pts), (0.50, pts)])
    params = [p for _, p in result.fitted_slices]
    assert verify_ssvi_calendar_free(params)


# ── Mutation-testing regressions (term_structure.py) ─────────────────
# These pin the exact constraint arithmetic and boundary semantics that
# the end-to-end fit tests only exercise through an interior optimum,
# so constraint-function mutations otherwise survive.
def test_butterfly_constraints_exact_values() -> None:
    """The four GJ butterfly residuals must equal their closed-form
    expressions exactly (pins the arb-check arithmetic, not just the
    fitted outcome).  rho != 0 so the (1+rho)/(1-rho) split is
    distinguishable.

    The first two residuals (condition 1, ``theta*psi*(1+|rho|) < 4``,
    STRICT in GJ Theorem 4.2) are shifted by
    ``_GJ_CONDITION1_STRICT_EPS`` so scipy's closed constraint sets
    reject an exact equality with the boundary; the last two (condition
    2, non-strict) are unshifted.  ``rtol=0`` pins the nudge — the
    default rtol would hide it."""
    for theta, rho, psi in [
        (0.5, -0.3, 1.2),
        (0.2, 0.6, 2.0),
    ]:
        res = _butterfly_constraints(theta, rho, psi)
        expected = np.array([
            4.0 - theta * psi * (1.0 + rho) - _GJ_CONDITION1_STRICT_EPS,
            4.0 - theta * psi * (1.0 - rho) - _GJ_CONDITION1_STRICT_EPS,
            4.0 - theta * psi * psi * (1.0 + rho),
            4.0 - theta * psi * psi * (1.0 - rho),
        ])
        assert np.allclose(res, expected, atol=1e-12, rtol=0)


def test_verify_hm_condition_empty_and_single_slice() -> None:
    """Degenerate inputs are arb-free by convention (no pair to check)."""
    assert verify_hm_condition([]) is True
    assert verify_hm_condition([SSVIParams(theta=0.1, rho=0.0, psi=1.0)]) is True
    assert verify_ssvi_calendar_free(None) is True
    assert verify_ssvi_calendar_free(
        [SSVIParams(theta=0.1, rho=0.0, psi=1.0)]
    ) is True


def test_verify_hm_condition_detects_two_slice_theta_dip() -> None:
    """A two-slice pair whose theta decreases must be rejected (kills
    mutations that only check the first slice or skip the n<=2 case)."""
    p1 = SSVIParams(theta=0.08, rho=-0.3, psi=0.5)
    p2 = SSVIParams(theta=0.03, rho=-0.2, psi=0.35)
    assert verify_hm_condition([p1, p2]) is False


def test_verify_hm_condition_detects_violation_on_last_pair() -> None:
    """A three-slice surface whose FINAL adjacent pair violates must be
    rejected (kills range(n-2) that drops the last pair)."""
    good = SSVIParams(theta=0.08, rho=-0.3, psi=0.5)
    dip = SSVIParams(theta=0.05, rho=-0.1, psi=0.5)
    assert verify_hm_condition([good, good, dip]) is False


def test_verify_hm_condition_ratio_boundary() -> None:
    """A pair with |ratio| strictly in (1, 2) must be rejected: kills the
    threshold loosening (> 1 -> > 2) and the denominator arithmetic
    mutations (chi[i+1]-chi[i] -> +, / denom -> * denom)."""
    p1 = SSVIParams(theta=1.0, rho=0.90, psi=1.0)
    p2 = SSVIParams(theta=1.05, rho=0.95, psi=1.0)
    # chi1=1.0, chi2=1.05, numerator=|0.95*1.05 - 0.90*1.0|=0.0975,
    # denom=0.05 => ratio=1.95 in (1, 2).
    assert verify_hm_condition([p1, p2]) is False


def test_verify_hm_condition_breakdown_schema() -> None:
    """The breakdown dicts must expose the documented keys and flag a
    ratio strictly above 1 (kills key renames and threshold loosening in
    the diagnostic output)."""
    p1 = SSVIParams(theta=1.0, rho=0.90, psi=1.0)
    p2 = SSVIParams(theta=1.05, rho=0.95, psi=1.0)
    rows = verify_hm_condition_breakdown(
        [(0.25, p1), (1.00, p2)], [None],
    )
    # Only slices with a valid predecessor get a row.
    assert len(rows) == 1
    row = rows[0]
    assert row["slice_T"] == 1.00
    assert set(row.keys()) == {
        "slice_T", "prev_T", "theta_self", "theta_prev", "theta_ok",
        "chi_self", "chi_prev", "chi_ok", "rho_chi_self",
        "rho_chi_prev", "ratio_value", "ratio_ok", "failing_conditions",
    }
    assert row["ratio_value"] == pytest.approx(1.95, abs=1e-6)
    assert row["ratio_ok"] is False
    assert "ratio" in row["failing_conditions"]


def test_fit_ssvi_surface_sequential_recovers_ground_truth() -> None:
    """The hard-constrained fit must recover the (feasible) ground-truth
    parameters, not merely produce a monotone surface.  This pins the
    optimizer bounds/objective: a mutation that breaks the variable
    bounds or the objective would move the optimum away from truth."""
    result = fit_ssvi_surface_sequential(_make_slices_data())
    assert result.fallback_slices == []
    fitted_by_T = dict(result.fitted_slices)
    for T, truth in zip(_EXPIRIES, _TRUTH):
        p = fitted_by_T[T]
        assert p.theta == pytest.approx(truth["theta"], rel=0.02)
        assert p.rho == pytest.approx(truth["rho"], abs=0.02)
        assert p.psi == pytest.approx(truth["psi"], rel=0.05)


# ── Mutation-testing regressions: seed block & fallback routing ──────
# The seed block in _fit_slice (the local `from scipy.optimize import
# least_squares as _ls` call, the theta0/rho0/p0 floors, the prev_chi
# reset and the rho clip) is exercised through an interior optimum by
# the end-to-end tests, so arithmetic mutations there survive.  These
# tests script the seed and the optimizer and pin the exact x0 the
# optimizer receives, plus the fallback-routing decisions downstream.
def test_fit_slice_seed_adjustment_pins_floor_clip_and_prev_chi(monkeypatch) -> None:
    """The seed block in ``_fit_slice`` — the ``_ls`` least-squares seed,
    its theta0/rho0/p0 floors, the ``prev_chi`` reset and the rho clip —
    must feed the optimizer exactly the documented ``x0``.

    This pins today's seed arithmetic at the seed level, so a seed tweak
    that would flip the infeasible-dip fallback (the m66 correctness
    concern) is caught here rather than only through the end-to-end fit.
    m45 (rho clip bound) and m54 (``prev_chi = prev.theta * prev.psi``)
    move the seed across a genuinely MARGINAL fallback decision — the
    constrained optimizer can converge or fall back either way.  m66's
    floor is different: ``max(p0, 1.000001)`` can suppress a fallback on
    a TRULY infeasible slice by starting the optimizer from a high-psi
    corner where the ``chi0 < prev_chi + eps_chi`` reset never fires, so
    it is the correctness-relevant one.

    Variant 1: seed p0 = 0.55 -> chi0 stays above prev_chi + eps_chi,
    so the floor and the clip fully determine x0 (p0 = 0.55, rho0 =
    -0.99, theta0 = 0.1).
    Variant 2: seed p0 = 0.01 -> chi0 < prev_chi + eps_chi triggers the
    prev_chi reset ``p0 = (prev_chi + eps_chi) / theta0``.
    """
    from scipy.optimize import OptimizeResult
    import arbfree_vol.ssvi.term_structure as ts

    points = _fit_points()
    prev = SSVIParams(theta=0.08, rho=-0.3, psi=0.5)
    recorded: list[np.ndarray] = []

    def _seed_ls(seed_x):
        def _fake_ls(fun, x0=None, bounds=None, **kwargs):
            return OptimizeResult(
                x=np.array(seed_x, dtype=np.float64),
                success=True,
                status=1,
                message="scripted seed",
            )
        return _fake_ls

    def _recording_minimize(fun, x0, method=None, bounds=None, constraints=None, options=None):
        recorded.append(np.array(x0, dtype=np.float64))
        return _opt_result(1, "Optimization terminated successfully", success=True)

    monkeypatch.setattr(ts, "minimize", _recording_minimize)

    # Variant 1: no prev_chi reset — the floor and the clip decide p0.
    monkeypatch.setattr("scipy.optimize.least_squares", _seed_ls([0.1, -0.995, 0.55]))
    ts._fit_slice(points, prev=prev)

    expected_theta0 = max(prev.theta + 1e-9, 0.1)          # 0.1
    expected_rho0 = float(np.clip(-0.995, -0.99, 0.99))    # -0.99 (kills m45)
    expected_p0 = max(0.55, 1e-6)                          # 0.55, no reset (kills m54, m66)
    expected_x0 = np.array([
        expected_theta0,
        float(np.arctanh(expected_rho0)),
        float(np.log(expected_p0)),
    ])
    assert len(recorded) == 1
    assert recorded[0] == pytest.approx(expected_x0, abs=1e-12)

    # Variant 2: seed p0 is small, so the prev_chi reset fires.
    recorded.clear()
    monkeypatch.setattr("scipy.optimize.least_squares", _seed_ls([0.1, -0.995, 0.01]))
    ts._fit_slice(points, prev=prev)

    expected_p0 = (prev.theta * prev.psi + 1e-6) / expected_theta0  # 0.40001
    expected_x0 = np.array([
        expected_theta0,
        float(np.arctanh(expected_rho0)),
        float(np.log(expected_p0)),
    ])
    assert len(recorded) == 1
    assert recorded[0] == pytest.approx(expected_x0, abs=1e-12)


def test_infeasible_slice_falls_back_regardless_of_high_seed(monkeypatch) -> None:
    """Even with an m66-style high seed (p0 = 1.000001), a genuinely
    infeasible slice must route into ``fallback_slices`` when the
    optimizer reports non-convergence, instead of being silently
    hard-certified.

    This pins the routing MECHANISM: the fallback decision must be
    driven by the optimizer's success flag, not by where the seed
    happens to land.  Both slices use a high seed and non-converged
    trust-constr + SLSQP results; the unconstrained per-slice fallback
    then recovers the true theta of each slice.
    """
    from scipy.optimize import OptimizeResult

    ts, fake_minimize = _scripted_minimize([
        _opt_result(4, "The maximum number of function evaluations is exceeded."),
        _opt_result(1, "Function evaluation required (f & c)"),
        _opt_result(4, "The maximum number of function evaluations is exceeded."),
        _opt_result(1, "Function evaluation required (f & c)"),
    ])
    monkeypatch.setattr(ts, "minimize", fake_minimize)

    def _high_seed_ls(fun, x0=None, bounds=None, **kwargs):
        return OptimizeResult(
            x=np.array([0.08, -0.3, 1.000001], dtype=np.float64),
            success=True,
            status=1,
            message="scripted high seed",
        )

    monkeypatch.setattr("scipy.optimize.least_squares", _high_seed_ls)

    ks = np.linspace(-1.0, 1.0, 9)
    slices_data = [
        (0.25, [(float(k), ssvi_w(float(k), 0.08, -0.3, 0.5)) for k in ks]),
        (0.50, [(float(k), ssvi_w(float(k), 0.03, -0.2, 0.35)) for k in ks]),
    ]

    result = fit_ssvi_surface_sequential(slices_data)

    # The theta-dipping slice must fall back, not be hard-certified.
    assert 0.50 in result.fallback_slices, (
        f"expected T=0.50 in fallback_slices, got {result.fallback_slices}"
    )
    assert result.failed_slices == []
    fitted_by_T = dict(result.fitted_slices)
    assert 0.50 in fitted_by_T, (
        f"expected T=0.50 in fitted_slices, got {result.fitted_slices}"
    )
    # The fallback params come from the unconstrained per-slice fit.
    assert fitted_by_T[0.50].theta == pytest.approx(0.03, rel=0.02)


def test_fallback_slice_fitted_slices_prev_keeps_predecessor(monkeypatch) -> None:
    """A fallback slice must keep its last hard-constrained predecessor
    in ``fitted_slices_prev`` (kills mutmut_39, which changes the
    fallback-branch ``append(prev_T_for_this_slice)`` to
    ``append(None)``).  The middle slice falls back; it must still
    record T=0.25 as its calibration predecessor rather than None.
    """
    import arbfree_vol.ssvi.term_structure as ts

    slices_data = _make_slices_data()

    # Monkeypatch _fit_slice to fail on the 2nd call (the middle slice).
    call_count = {"n": 0}
    _real_fit_slice = ts._fit_slice

    def _failing_fit_slice(points, prev=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated infeasible H&M constraints")
        return _real_fit_slice(points, prev=prev, **kwargs)

    monkeypatch.setattr(ts, "_fit_slice", _failing_fit_slice)

    result = fit_ssvi_surface_sequential(slices_data)

    assert 0.50 in result.fallback_slices
    assert len(result.fitted_slices_prev) == len(result.fitted_slices)
    assert result.fitted_slices_prev[1] == 0.25, (
        "fallback slice T=0.50 must keep its last hard-constrained "
        "predecessor (T=0.25) in fitted_slices_prev, not None: "
        f"got {result.fitted_slices_prev}"
    )
    assert result.fitted_slices_prev[1] is not None


def test_hard_fit_within_eps_of_boundary_not_silently_certified(monkeypatch) -> None:
    """Desired invariant (m66, now a hard regression tripwire): a hard-fit
    pinned within eps of the H&M Prop 3.1 boundary (the m66 degenerate
    corner) must be routed to fallback, not silently certified arb-free.

    Deterministic: `ts._fit_slice` is scripted to land on the measured m66
    corner for the theta-dipping slice, `ts.fit_ssvi_slice` is scripted to
    the true dip params for the RMSE baseline, and the real
    `fit_ssvi_surface_sequential` routing logic runs.  Formerly xfail
    (``docs/code_review_findings.md`` §6.7): before the post-fit margin
    check landed (``ssvi/_hm_margin.py``) this test failed — the corner
    was accepted as hard; after the fix it began XPASSing.  The marker is
    now removed: a regression of the margin check fails the suite red.
    """
    import arbfree_vol.ssvi.term_structure as ts

    # m66 measured corner (docs/code_review_findings.md §6.7), pinned to its prev.
    prev_params = SSVIParams(theta=0.1192518709, rho=0.08325035, psi=0.47564518)
    corner = SSVIParams(theta=0.119251872, rho=0.083266515, psi=0.475653565)
    truth2 = SSVIParams(theta=0.07, rho=0.2, psi=0.55)

    ks = np.linspace(-1.0, 1.0, 9)
    slices_data = [
        (0.25, [(float(k), ssvi_w(float(k), 0.08, -0.3, 0.5)) for k in ks]),
        (1.00, [(float(k), ssvi_w(float(k), 0.12, 0.1, 0.4)) for k in ks]),
        (2.00, [(float(k), ssvi_w(float(k), 0.07, 0.2, 0.55)) for k in ks]),
    ]

    real_fit_slice = ts._fit_slice

    def _scripted_fit_slice(points, prev=None, **kwargs):
        k0, w0 = points[0]
        if abs(w0 - ssvi_w(k0, 0.07, 0.2, 0.55)) < 1e-8:
            return corner        # T=2.0 dip slice -> m66 corner
        if abs(w0 - ssvi_w(k0, 0.12, 0.1, 0.4)) < 1e-8:
            return prev_params   # T=1.0 slice -> the prev that pins the corner
        return real_fit_slice(points, prev=prev, **kwargs)

    monkeypatch.setattr(ts, "_fit_slice", _scripted_fit_slice)
    monkeypatch.setattr(ts, "fit_ssvi_slice", lambda points: truth2)

    result = fit_ssvi_surface_sequential(slices_data)

    assert 2.00 in result.fallback_slices, (
        "the m66 corner hard-fit must be routed to fallback, not silently certified"
    )


def test_degenerate_corner_with_failed_baseline_routes_to_fallback(monkeypatch) -> None:
    """A hard fit pinned within the H&M boundary window must be routed
    to fallback even when the unconstrained baseline fit fails.

    Scripts the m66 corner parameters (theta_delta ~1e-9, chi_delta
    ~1e-6, ratio ~0.9998) AND makes the baseline ``fit_ssvi_slice`` call
    inside the degenerate-corner check raise.  Pre-fix,
    ``_hard_fit_is_degenerate_corner`` returned False when the baseline
    raised ("conservative"), silently certifying the boundary-adjacent
    hard fit.  Post-fix the boundary proximity alone flags the corner,
    the caller raises RuntimeError, and the honest fallback recovers the
    true dip params — so the slice lands in ``fallback_slices``.
    """
    import arbfree_vol.ssvi.term_structure as ts

    # m66 measured corner (docs/code_review_findings.md §6.7), pinned to its prev.
    prev_params = SSVIParams(theta=0.1192518709, rho=0.08325035, psi=0.47564518)
    corner = SSVIParams(theta=0.119251872, rho=0.083266515, psi=0.475653565)
    truth2 = SSVIParams(theta=0.07, rho=0.2, psi=0.55)

    ks = np.linspace(-1.0, 1.0, 9)
    slices_data = [
        (0.25, [(float(k), ssvi_w(float(k), 0.08, -0.3, 0.5)) for k in ks]),
        (1.00, [(float(k), ssvi_w(float(k), 0.12, 0.1, 0.4)) for k in ks]),
        (2.00, [(float(k), ssvi_w(float(k), 0.07, 0.2, 0.55)) for k in ks]),
    ]

    real_fit_slice = ts._fit_slice

    def _scripted_fit_slice(points, prev=None, **kwargs):
        k0, w0 = points[0]
        if abs(w0 - ssvi_w(k0, 0.07, 0.2, 0.55)) < 1e-8:
            return corner        # T=2.0 dip slice -> m66 corner
        if abs(w0 - ssvi_w(k0, 0.12, 0.1, 0.4)) < 1e-8:
            return prev_params   # T=1.0 slice -> the prev that pins the corner
        return real_fit_slice(points, prev=prev, **kwargs)

    calls = {"n": 0}

    def _baseline_fails_then_recovers(points):
        # First call is the RMSE baseline inside the degenerate-corner
        # check for T=2.0; it fails.  The fallback call after the corner
        # is flagged must succeed so the slice is recorded as a fallback.
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("unconstrained baseline fit failed")
        return truth2

    monkeypatch.setattr(ts, "_fit_slice", _scripted_fit_slice)
    monkeypatch.setattr(ts, "fit_ssvi_slice", _baseline_fails_then_recovers)

    result = fit_ssvi_surface_sequential(slices_data)

    assert 2.00 in result.fallback_slices, (
        "a boundary-window hard fit must be routed to fallback even when "
        f"the baseline fit fails; got fallback_slices={result.fallback_slices}"
    )
    assert 2.00 not in result.failed_slices, (
        f"the fallback must succeed; got failed_slices={result.failed_slices}"
    )
    fitted_by_T = dict(result.fitted_slices)
    assert 2.00 in fitted_by_T
    # The fallback recovers the true dip params.
    assert fitted_by_T[2.00].theta == pytest.approx(0.07, rel=0.02)


def test_fit_slice_raises_on_too_few_points() -> None:
    """``_fit_slice`` enforces the 5-point minimum before optimizing."""
    import arbfree_vol.ssvi.term_structure as ts

    pts = [(float(k), 0.04) for k in [-1.0, 0.0, 1.0]]  # only 3 points
    with pytest.raises(ValueError, match="at least 5 points"):
        ts._fit_slice(pts)
