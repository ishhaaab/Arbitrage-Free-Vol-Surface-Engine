"""Tests for the eSSVI sequential term-structure fitter.

Verifies the Hendriks & Martini (2019) Prop 3.1 no-calendar-spread
conditions hold after calibration.

Reference: Hendriks & Martini (2019), "The Extended SSVI Volatility
Surface", J. Comput. Finance 22(5), 25-39, Prop 3.1.
"""

import numpy as np
import pytest
from math import sqrt

from arbfree_vol.ssvi.model import SSVIParams, ssvi_w, to_raw_svi_params
from arbfree_vol.ssvi.term_structure import (
    fit_ssvi_surface_sequential,
    verify_hm_condition,
    SequentialFitResult,
)
from arbfree_vol.arbitrage.svi_detect import detect_svi_surface


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


# ── Test 1 ──────────────────────────────────────────────────────────
def test_theta_and_chi_non_decreasing() -> None:
    """After calibration, theta_i and chi_i=theta_i*psi_i must be
    non-decreasing across slices (Hendriks & Martini 2019, Prop 3.1,
    conditions (a) and (b))."""
    slices_data = _make_slices_data()
    result = fit_ssvi_surface_sequential(slices_data)
    params = [p for _, p in result.fitted_slices]

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
    slices_data = _make_slices_data()
    result = fit_ssvi_surface_sequential(slices_data)
    params = [p for _, p in result.fitted_slices]

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
    slices_data = _make_slices_data()
    result = fit_ssvi_surface_sequential(slices_data)
    params = [p for _, p in result.fitted_slices]

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

    truth1 = dict(theta=0.04, rho=-0.3, psi=0.5)
    truth2 = dict(theta=0.08, rho=-0.2, psi=0.6)
    truth3 = dict(theta=0.14, rho=-0.1, psi=0.65)
    ks = np.linspace(-1.0, 1.0, 9).tolist()

    def _pts(truth):
        return [
            (float(k), ssvi_w(float(k), truth["theta"], truth["rho"], truth["psi"]))
            for k in ks
        ]

    slices_data = [
        (0.25, _pts(truth1)),
        (0.50, _pts(truth2)),
        (1.00, _pts(truth3)),
    ]

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

    print(f"  result expiries: {[T for T, _ in result.fitted_slices]}")
    print(f"  fallback_slices: {result.fallback_slices}")
    print(f"  failed_slices: {result.failed_slices}")
    for i, (T, p) in enumerate(result.fitted_slices):
        print(f"  slice {i} (T={T}): theta={p.theta:.6f}, rho={p.rho:.4f}, psi={p.psi:.4f}")

    # verify_hm_condition may be False because of the fallback
    params_only = [p for _, p in result.fitted_slices]
    hm_ok = verify_hm_condition(params_only)
    print(f"  verify_hm_condition: {hm_ok}")
    # We don't assert True or False — just that the function returns
    # without raising.


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

    truth1 = dict(theta=0.04, rho=-0.3, psi=0.5)
    truth2 = dict(theta=0.08, rho=-0.2, psi=0.6)
    truth3 = dict(theta=0.14, rho=-0.1, psi=0.65)
    ks = np.linspace(-1.0, 1.0, 9).tolist()

    def _pts(truth):
        return [
            (float(k), ssvi_w(float(k), truth["theta"], truth["rho"], truth["psi"]))
            for k in ks
        ]

    slices_data = [
        (0.25, _pts(truth1)),
        (0.50, _pts(truth2)),
        (1.00, _pts(truth3)),
    ]

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


def test_fit_slice_slsqp_success_status_accepted(monkeypatch) -> None:
    """A converged fit (success=True, status 1/2 for trust-constr) still
    returns fitted parameters."""
    from math import exp

    ts, fake_minimize = _scripted_minimize([
        _opt_result(1, "Optimization terminated successfully", success=True),
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
_DIP_TRUTH = [
    (0.25, dict(theta=0.08, rho=-0.3, psi=0.5)),
    (0.50, dict(theta=0.03, rho=-0.2, psi=0.35)),  # theta + chi dip
    (1.00, dict(theta=0.12, rho=0.1, psi=0.4)),
    (2.00, dict(theta=0.07, rho=0.2, psi=0.55)),   # theta dip again
]


def _dip_slices_data(with_tail: bool = False):
    """Build (expiry, [(k, w)]) data from the dip ground truth."""
    data = [
        (T, [(float(k), ssvi_w(float(k), t["theta"], t["rho"], t["psi"])) for k in _KS])
        for T, t in _DIP_TRUTH
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
    """
    result = fit_ssvi_surface_sequential(_dip_slices_data())

    assert result.failed_slices == []
    params_only = [p for _, p in result.fitted_slices]
    hm_ok = verify_hm_condition(params_only)
    if not hm_ok:
        assert len(result.fallback_slices) > 0, (
            "verify_hm_condition reports a violation but no slice is "
            "recorded in fallback_slices — the H&M hard constraints "
            "appear to be missing from _fit_slice"
        )


def test_fitted_slices_prev_threads_last_hard_constrained_predecessor() -> None:
    """fitted_slices_prev must point at the last HARD-CONSTRAINED slice
    before each fitted slice, skipping fallback slices.

    On the current dataset slice T=2.0 really falls back (the
    'Positive directional derivative for linesearch' failure also seen
    on live data), giving prev=[None, 0.25, 0.5, 1.0, 1.0]: the T=3.0
    slice anchors to T=1.0, NOT to the fallback slice T=2.0.  A
    regression where a fallback slice wrongly becomes the predecessor
    of the next slice would make the recomputed expectation differ.
    """
    result = fit_ssvi_surface_sequential(_dip_slices_data(with_tail=True))

    assert len(result.fitted_slices_prev) == len(result.fitted_slices)
    assert result.fitted_slices_prev[0] is None

    expected_prev: list[float | None] = []
    last_valid_T: float | None = None
    for T, _ in result.fitted_slices:
        expected_prev.append(last_valid_T)
        if T not in result.fallback_slices:
            last_valid_T = T

    assert result.fitted_slices_prev == expected_prev, (
        "fitted_slices_prev threading drifted from the documented "
        "last-hard-constrained-predecessor rule:\n"
        f"  got      {result.fitted_slices_prev}\n"
        f"  expected {expected_prev}"
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
    from arbfree_vol.ssvi.term_structure import verify_ssvi_calendar_free

    flat_w = 0.04
    pts = [(float(k), flat_w) for k in np.linspace(-1.0, 1.0, 9)]
    result = fit_ssvi_surface_sequential([(0.25, pts), (0.50, pts)])
    params = [p for _, p in result.fitted_slices]
    assert verify_ssvi_calendar_free(params)
