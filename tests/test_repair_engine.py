"""Tests for the repair engine.

The repair pipeline (`arbfree_vol.repair.engine.repair`) dispatches to one of
three model paths; this file covers all three plus the cross-cutting slice
bookkeeping:

- raw SVI (default): per-slice constrained calibration — ``test_repair_svi_*``
  and the clean-surface tests at the top.
- eSSVI (``use_ssvi=True``): sequential H&M calendar-arb-free fit — the
  ``test_repair_essvi_*`` tests and the ``_DIP_TRUTH_ENGINE`` theta-dip
  fixtures (degenerate-corner routing / fallback / m66 tripwire).
- SABR (``use_sabr=True``): B-spline term-structure fit mapped to raw SVI —
  the ``test_sabr_*`` tests and the SABR->SVI mapping bookkeeping.

Cross-cutting slice bookkeeping (exercised for all three paths):

- ``failed_slices`` / ``fallback_slices``: per-slice fit failure vs. an honest
  unconstrained fallback.
- ``sabr_mapping_failed_slices``: SABR->SVI mapping failures (RuntimeError /
  ValueError) — every slice accounted for, never silently dropped.
- no-forward-estimate slices: a slice with no ``estimate_forward_curve`` entry
  is recorded in ``failed_slices``, not silently skipped.

Shared constants and surface-builder helpers live in
``tests/repair_helpers.py``.
"""
import logging
import pytest
from pytest import approx

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.repair.engine import repair
from arbfree_vol.sabr.model import SABRParams

from tests.repair_helpers import (
    SPOT, R, Q, T, _DUMMY_DATE,
    _DIP_TRUTH_ENGINE, _SVI_TRUTH_ENGINE,
    _bs_price, _clean_surface,
    _ssvi_priced_surface, _svi_priced_surface,
    _flat_bs_surface, _forward_curve_missing,
)


def test_repair_clean_surface_rejects_nothing() -> None:
    surface = _clean_surface(n_strikes=7)

    report = repair(surface)

    assert report.metrics.n_rejected == 0
    assert report.metrics.n_slices_input == 1
    assert report.metrics.n_slices_fitted == 1
    assert report.metrics.n_violations_before == 0
    assert report.metrics.n_violations_after == 0
    assert len(report.fitted_slices) == 1
    assert report.cleaned_surface is not None
    assert len(report.cleaned_surface.slices[0].quotes) == 14


def test_repair_rejects_monotonicity_violation() -> None:
    # A clean slice + one extra call with a higher price at a higher strike
    # (= monotonicity violation). The bad quote should be rejected.
    clean_quotes: list[Quote] = []
    for step in range(-3, 4):  # 7 strikes
        K = SPOT + step * 10.0
        clean_quotes.append(
            Quote(strike=K, option_type=OptionType.CALL,
                  price=_bs_price(OptionType.CALL, K))
        )
        clean_quotes.append(
            Quote(strike=K, option_type=OptionType.PUT,
                  price=_bs_price(OptionType.PUT, K))
        )
    # Bad quote: strike=110 with price=20 (should be ~5)
    clean_quotes.append(
        Quote(strike=110.0, option_type=OptionType.CALL, price=20.0)
    )

    surface = VolSurface(
        spot=SPOT, risk_free=R, div_yield=Q,
        slices=[ExpirySlice(expiry_time=T, quotes=clean_quotes)],
    )

    report = repair(surface)

    assert report.metrics.n_rejected >= 1
    assert report.metrics.n_slices_fitted == 1
    # The bad quote at K=110 call should be in the rejected list
    assert any(
        r.strike == 110.0 and r.option_type == OptionType.CALL
        for r in report.rejected
    )


def test_repair_surface_with_too_few_quotes() -> None:
    # Only 2 quotes per slice — not enough for SVI (need >=5).
    surface = VolSurface(
        spot=SPOT, risk_free=R, div_yield=Q,
        slices=[ExpirySlice(
            expiry_time=T,
            quotes=[
                Quote(strike=100.0, option_type=OptionType.CALL, price=10.0),
                Quote(strike=110.0, option_type=OptionType.CALL, price=5.0),
            ],
        )],
    )

    report = repair(surface)

    assert report.metrics.n_slices_fitted == 0
    assert len(report.fitted_slices) == 0


def test_repair_multiple_slices() -> None:
    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    quotes1: list[Quote] = []
    quotes2: list[Quote] = []
    for K in strikes:
        for otype in [OptionType.CALL, OptionType.PUT]:
            quotes1.append(
                Quote(strike=K, option_type=otype,
                      price=_bs_price(otype, K, tt=0.5))
            )
            quotes2.append(
                Quote(strike=K, option_type=otype,
                      price=_bs_price(otype, K, tt=1.0))
            )

    surface = VolSurface(
        spot=SPOT, risk_free=R, div_yield=Q,
        slices=[
            ExpirySlice(expiry_time=0.5, quotes=quotes1),
            ExpirySlice(expiry_time=1.0, quotes=quotes2),
        ],
    )

    report = repair(surface)

    assert report.metrics.n_rejected == 0
    assert report.metrics.n_slices_fitted == 2
    assert len(report.fitted_slices) == 2


def test_repair_metrics_consistency() -> None:
    surface = _clean_surface(n_strikes=7)

    report = repair(surface)

    m = report.metrics
    assert m.n_total_quotes == 14
    assert m.n_rejected + m.n_violations_before == 0
    assert m.n_slices_fitted <= m.n_slices_input
    assert 0.0 <= m.rejection_rate <= 1.0


def test_repair_with_ssvi_populates_fitted_ssvi_slices() -> None:
    surface = _clean_surface(n_strikes=7)

    report = repair(surface, use_ssvi=True)

    assert report.metrics.n_slices_fitted == 1
    assert len(report.fitted_slices) == 1
    assert len(report.fitted_ssvi_slices) == 1
    # The raw SVI params (mapped from eSSVI) are also present so the
    # existing SVI-based pipeline (plots, detection) keeps working.
    assert report.fitted_slices[0].params is not None
    # The native eSSVI parameters carry theta, rho, psi.
    fssvi = report.fitted_ssvi_slices[0]
    assert fssvi.ssvi.theta > 0
    assert -1.0 < fssvi.ssvi.rho < 1.0
    assert fssvi.ssvi.psi > 0


def test_repair_default_omits_ssvi() -> None:
    # Default path (use_ssvi=False) should not populate fitted_ssvi_slices.
    surface = _clean_surface(n_strikes=7)

    report = repair(surface)

    assert report.fitted_ssvi_slices == ()


def test_repair_with_sabr_populates_fitted_sabr_slices() -> None:
    surface = _clean_surface(n_strikes=7)

    report = repair(surface, use_sabr=True)

    assert report.metrics.n_slices_fitted == 1
    assert len(report.fitted_slices) == 1
    assert len(report.fitted_sabr_slices) == 1
    # The raw SVI params (mapped from SABR) should be present
    assert report.fitted_slices[0].params is not None
    # The native SABR parameters
    fsabr = report.fitted_sabr_slices[0]
    assert fsabr.sabr.alpha > 0
    assert fsabr.sabr.nu > 0
    assert -1.0 < fsabr.sabr.rho < 1.0


def test_repair_with_sabr_and_ssvi_mutually_exclusive() -> None:
    surface = _clean_surface(n_strikes=7)

    with pytest.raises(ValueError):
        repair(surface, use_ssvi=True, use_sabr=True)


def test_repair_sabr_then_build_fitted_surface_then_iv_at() -> None:
    """SABR repair -> build_fitted_surface -> iv_at round-trip."""
    from arbfree_vol.surface.interpolate import build_fitted_surface, iv_at

    surface = _clean_surface(n_strikes=7)
    report = repair(surface, use_sabr=True)

    assert len(report.fitted_sabr_slices) == 1
    assert report.metrics.n_slices_fitted == 1

    fs = build_fitted_surface(report)
    assert len(fs.fitted_slices) == 1

    # iv_at at the exact slice expiry should return a plausible vol
    iv = iv_at(fs, K=SPOT, T=T)
    assert 0.05 < iv < 1.0

    # T below the single slice should raise
    with pytest.raises(ValueError):
        iv_at(fs, K=SPOT, T=0.01)

    # T above the single slice should raise
    with pytest.raises(ValueError):
        iv_at(fs, K=SPOT, T=5.0)


def test_repair_constrained_calibration_leaves_no_butterfly_violations() -> None:
    """With constrained SVI calibration, a clean flat-vol surface must
    produce a fitted surface that is entirely free of arbitrage."""
    surface = _clean_surface(n_strikes=7)

    report = repair(surface)

    assert report.metrics.n_slices_fitted == 1
    assert report.remaining_violations.is_arbitrage_free, (
        "Constrained calibration should produce an arb-free fit "
        "on clean input data"
    )


def test_repair_svi_path_fixes_calendar_violation() -> None:
    """The SVI repair path must prevent cross-slice calendar arbitrage.

    Build a synthetic surface from two SVI truth parameter sets chosen
    so the short-dated slice's wings exceed the long-dated slice's
    wings (provably non-calendar per detect_svi_surface). Then run
    repair() and assert the fitted surface is calendar-consistent.

    Pre-fix: the per-slice SVI fits ignore cross-slice ordering, so
    the post-repair detect_svi_surface reports a CALENDAR violation.
    Post-fix: the calendar penalty in calibrate_constrained forces
    w_current(k) >= w_prev(k) on the k-grid, and the post-repair
    detect_svi_surface is clean.
    """
    from math import exp, sqrt as _sqrt
    from arbfree_vol.svi.model import SVIParams, svi_total_variance
    from arbfree_vol.arbitrage.svi_detect import detect_svi_surface

    # NOTE: these truth params are synthetic and chosen purely so the
    # short-dated slice's wings exceed the long-dated slice's wings
    # (calendar arbitrage). The implied short-dated ATM vol is far
    # above any real equity. This is a regression test for the
    # repair path's calendar penalty, not a realistic data scenario.
    long_params = SVIParams(a=0.04, b=0.08, rho=-0.3, m=0.0, sigma=0.4)
    short_params = SVIParams(a=0.01, b=0.30, rho=-0.5, m=0.0, sigma=0.3)

    # ---- Meta-check: the truth params themselves must be non-calendar.
    # If this fails, retune the params above. ----
    truth_report = detect_svi_surface([
        (1.0, long_params),
        (0.02, short_params),
    ])
    cal_violations = [
        v for v in truth_report.violations
        if v.kind.value == "calendar"
    ]
    assert cal_violations, (
        "Test setup error: truth params do not produce a calendar "
        "violation on detect_svi_surface. Re-tune short_params so the "
        "short-dated slice's wings exceed the long-dated slice's wings "
        "for some k in [-1.5, 1.5]."
    )

    # ---- Generate quotes from the truth SVI total variances. ----
    ks = [-1.5 + 3.0 * i / 20.0 for i in range(21)]
    F = 100.0
    T_long, T_short = 1.0, 0.02

    def make_slice(T, params):
        quotes: list[Quote] = []
        for k in ks:
            K = F * exp(k)
            w = svi_total_variance(k, params.a, params.b,
                                   params.rho, params.m, params.sigma)
            sigma = _sqrt(w / T)
            quotes.append(Quote(strike=K, option_type=OptionType.CALL,
                                price=_bs_price(OptionType.CALL, K, sigma=sigma, tt=T)))
            quotes.append(Quote(strike=K, option_type=OptionType.PUT,
                                price=_bs_price(OptionType.PUT, K, sigma=sigma, tt=T)))
        return ExpirySlice(expiry_time=T, quotes=quotes)

    surface = VolSurface(
        spot=SPOT, risk_free=R, div_yield=Q,
        slices=[
            make_slice(T_long, long_params),
            make_slice(T_short, short_params),
        ],
    )

    report = repair(surface)

    assert report.metrics.n_slices_fitted == 2, (
        f"expected 2 fitted slices, got {report.metrics.n_slices_fitted}"
    )

    # ---- The post-fit surface must be calendar-consistent. ----
    svi_pairs = [(fs.expiry_time, fs.params) for fs in report.fitted_slices]
    fitted_report = detect_svi_surface(svi_pairs)
    fitted_cal = [
        v for v in fitted_report.violations
        if v.kind.value == "calendar"
    ]
    assert not fitted_cal, (
        f"SVI repair path left {len(fitted_cal)} calendar violation(s) "
        f"after fix; expected zero. Violations: "
        f"{[v.detail for v in fitted_cal]}"
    )
    assert fitted_report.is_arbitrage_free, (
        f"SVI repair path produced non-arb-free surface. All violations: "
        f"{[v.kind.value for v in fitted_report.violations]}"
    )


def test_repair_essvi_sequential_is_calendar_arb_free() -> None:
    """The eSSVI sequential fit must produce a calendar-arb-free surface.

    Build a 3-slice surface from flat BS vol 0.2 at expiries
    0.25, 0.5, 1.0.  Run repair(use_ssvi=True).  Assert:
    - 3 slices fitted
    - 3 fitted_ssvi_slices
    - 0 violations after
    - repair_infeasible is False
    - theta strictly increasing
    - chi = theta*psi strictly increasing
    """
    from math import exp

    expiries = [0.25, 0.5, 1.0]
    n_strikes = 7
    strikes = [SPOT * (1 + 0.1 * (i - n_strikes // 2)) for i in range(n_strikes)]

    slices: list[ExpirySlice] = []
    for T in expiries:
        quotes: list[Quote] = []
        for K in strikes:
            quotes.append(
                Quote(strike=K, option_type=OptionType.CALL,
                      price=_bs_price(OptionType.CALL, K, sigma=0.2, tt=T))
            )
            quotes.append(
                Quote(strike=K, option_type=OptionType.PUT,
                      price=_bs_price(OptionType.PUT, K, sigma=0.2, tt=T))
            )
        slices.append(ExpirySlice(expiry_time=T, quotes=quotes))

    surface = VolSurface(spot=SPOT, risk_free=R, div_yield=Q, slices=slices)

    report = repair(surface, use_ssvi=True)

    assert report.metrics.n_slices_fitted == 3, (
        f"expected 3 fitted slices, got {report.metrics.n_slices_fitted}"
    )
    assert len(report.fitted_ssvi_slices) == 3, (
        f"expected 3 fitted_ssvi_slices, got {len(report.fitted_ssvi_slices)}"
    )
    assert report.metrics.n_violations_after == 0, (
        f"expected 0 violations after, got {report.metrics.n_violations_after}"
    )
    assert report.repair_infeasible is False, (
        "repair_infeasible should be False for a clean surface"
    )

    # theta strictly increasing
    thetas = [s.ssvi.theta for s in report.fitted_ssvi_slices]
    for i in range(len(thetas) - 1):
        assert thetas[i + 1] > thetas[i], (
            f"theta not strictly increasing: {thetas}"
        )

    # chi = theta * psi strictly increasing
    chis = [s.ssvi.theta * s.ssvi.psi for s in report.fitted_ssvi_slices]
    for i in range(len(chis) - 1):
        assert chis[i + 1] > chis[i], (
            f"chi not strictly increasing: {chis}"
        )


@pytest.mark.slow
# Slow: runs the full SABR term-structure fit + constrained calibration on
# 3 slices (real optimizer work, ~9s).
def test_repair_sabr_term_structure_reduces_violations() -> None:
    """SABR term-structure path produces valid params on a 3-slice surface.

    Build a 3-slice surface (expiries 0.25, 0.5, 1.0) priced with flat
    BS vol 0.2.  Run repair(use_sabr=True).  Assert:
    - 3 slices fitted
    - 3 fitted_sabr_slices
    - Every SABRParams has alpha > 0, nu > 0, rho in (-1,1), beta == 0.5
    - Calendar violation count is small (<= 5)
    """
    from arbfree_vol.sabr.term_structure import EPS_FLOOR
    from arbfree_vol.arbitrage.report import ViolationType

    expiries = [0.25, 0.5, 1.0]
    n_strikes = 7
    strikes = [SPOT * (1 + 0.1 * (i - n_strikes // 2)) for i in range(n_strikes)]

    slices: list[ExpirySlice] = []
    for T_val in expiries:
        quotes: list[Quote] = []
        for K in strikes:
            quotes.append(
                Quote(strike=K, option_type=OptionType.CALL,
                      price=_bs_price(OptionType.CALL, K, sigma=0.2, tt=T_val))
            )
            quotes.append(
                Quote(strike=K, option_type=OptionType.PUT,
                      price=_bs_price(OptionType.PUT, K, sigma=0.2, tt=T_val))
            )
        slices.append(ExpirySlice(expiry_time=T_val, quotes=quotes))

    surface = VolSurface(spot=SPOT, risk_free=R, div_yield=Q, slices=slices)
    report = repair(surface, use_sabr=True)

    assert report.metrics.n_slices_fitted == 3, (
        f"expected 3 fitted slices, got {report.metrics.n_slices_fitted}"
    )
    assert len(report.fitted_sabr_slices) == 3, (
        f"expected 3 fitted_sabr_slices, got {len(report.fitted_sabr_slices)}"
    )

    # Every SABR param must be in valid range
    for fsabr in report.fitted_sabr_slices:
        p = fsabr.sabr
        assert p.alpha > EPS_FLOOR, f"alpha={p.alpha} not > EPS_FLOOR"
        assert p.nu > 0, f"nu={p.nu} not > 0"
        assert -1.0 < p.rho < 1.0, f"rho={p.rho} out of range"
        assert p.beta == 0.5

    # Calendar violations should be small (empirical path)
    cal_violations = [
        v for v in report.remaining_violations.violations
        if v.kind == ViolationType.CALENDAR
    ]
    assert len(cal_violations) <= 5, (
        f"too many calendar violations: {len(cal_violations)}"
    )


def test_repair_essvi_handles_infeasible_slice_gracefully(monkeypatch) -> None:
    """eSSVI repair must not crash when a slice's data makes the H&M
    hard constraints infeasible.  The fitter should fall back to the
    unconstrained per-slice fit for that slice, and the repair report
    should honestly flag ``repair_infeasible=True``.

    Build a 3-slice surface (flat vol 0.2) and monkeypatch
    ``_fit_slice`` in ``term_structure`` to raise ``RuntimeError`` on
    the second call, simulating an infeasible H&M constraint for the
    middle slice.
    """
    import arbfree_vol.ssvi.term_structure as ts

    expiries = [0.25, 0.5, 1.0]
    n_strikes = 7
    strikes = [SPOT * (1 + 0.1 * (i - n_strikes // 2)) for i in range(n_strikes)]

    slices: list[ExpirySlice] = []
    for T in expiries:
        quotes: list[Quote] = []
        for K in strikes:
            quotes.append(
                Quote(strike=K, option_type=OptionType.CALL,
                      price=_bs_price(OptionType.CALL, K, sigma=0.2, tt=T))
            )
            quotes.append(
                Quote(strike=K, option_type=OptionType.PUT,
                      price=_bs_price(OptionType.PUT, K, sigma=0.2, tt=T))
            )
        slices.append(ExpirySlice(expiry_time=T, quotes=quotes))

    surface = VolSurface(spot=SPOT, risk_free=R, div_yield=Q, slices=slices)

    # Monkeypatch _fit_slice to fail on the 2nd call
    call_count = {"n": 0}
    _real_fit_slice = ts._fit_slice

    def _failing_fit_slice(points, prev=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated infeasible H&M constraints")
        return _real_fit_slice(points, prev=prev, **kwargs)

    monkeypatch.setattr(ts, "_fit_slice", _failing_fit_slice)

    # Must not raise
    report = repair(surface, use_ssvi=True)

    # All 3 slices must have a fit (hard-constrained or fallback)
    assert len(report.fitted_ssvi_slices) == 3, (
        f"expected 3 fitted_ssvi_slices, got {len(report.fitted_ssvi_slices)}"
    )
    assert report.metrics.n_slices_fitted == 3, (
        f"expected 3 fitted slices, got {report.metrics.n_slices_fitted}"
    )
    # The fallback slice violates H&M, so repair_infeasible must be True
    assert report.repair_infeasible is True, (
        "repair_infeasible should be True when a slice fell back"
    )

    # The middle slice (T=0.5) should be in fallback_slices
    assert 0.5 in report.fallback_slices, (
        f"expected T=0.5 in fallback_slices, got {report.fallback_slices}"
    )
    # No slices should have failed entirely
    assert report.failed_slices == [], (
        f"expected empty failed_slices, got {report.failed_slices}"
    )


# ── Real (non-monkeypatched) honest fallback — xfail until m66 lands ─
# Same genuinely H&M-incompatible-at-face-value ground truth as the
# term-structure tests, but priced into a VolSurface so the FULL
# repair() pipeline runs: cleaning -> forward curve -> sequential eSSVI
# fit -> arb verification.  Pre-fix, the T=2.0 hard fit honestly fell
# back (H&M linesearch failure).  The strict Gatheral-Jacquier
# condition-1 eps shift re-routed SLSQP to a degenerate boundary corner
# that is silently certified arb-free — the SAME margin-check gap as
# mutmut_66 (docs/code_review_findings.md §6.7).  The test is xfail
# until a post-fit margin check lands and must NOT bless that corner.


@pytest.mark.xfail(
    reason="m66 margin-check gap (docs/code_review_findings.md §6.7, 2026-08-09): "
    "the strict Gatheral-Jacquier condition-1 eps shift changed SLSQP routing so "
    "this genuinely-unsatisfiable T=2.0 dip converges to a degenerate corner pinned "
    "at the H&M eps floors (theta_delta=eps_theta, chi_delta=eps_chi, ratio~0.9998, "
    "RMSE~0.05) and is silently certified arb-free. A post-fit margin check must "
    "route such fits to fallback_slices or set repair_infeasible; this test then "
    "flips to XPASS.",
    strict=False,
)
def test_repair_essvi_real_fallback_on_incompatible_data() -> None:
    """End-to-end: genuinely H&M-incompatible-at-face-value data through
    repair() honestly falls back (no monkeypatch) and the fallback is
    flagged — currently xfail-marked.

    The fixture is the same m66 dataset as docs/code_review_findings.md
    §6.7 (the theta-dip ground truth at T=0.5 and T=2.0).  Under the
    pre-fix code the T=2.0 hard-constrained fit failed the H&M
    linesearch and honestly fell back to the unconstrained fit — this
    test's original purpose: ``2.0 in report.fallback_slices``,
    ``repair_infeasible=True``, and the remaining calendar violation
    surfaced in ``n_violations_after``.  Under the strictness fix
    (``_GJ_CONDITION1_STRICT_EPS``) the optimizer now certifies the
    T=2.0 dip as a degenerate corner pinned at the H&M eps floors — the
    SAME underlying margin-check gap as mutmut_66 surfacing through a
    different mutation, not a genuinely feasible fit.  The test is
    therefore xfail until the post-fit margin check lands, and must NOT
    be used to bless the corner as correct.
    """
    report = repair(_ssvi_priced_surface(_DIP_TRUTH_ENGINE), use_ssvi=True)

    assert report.metrics.n_slices_fitted == 4, (
        f"expected 4 fitted slices, got {report.metrics.n_slices_fitted}"
    )
    assert len(report.fitted_ssvi_slices) == 4
    assert 2.0 in report.fallback_slices, (
        f"expected a real fallback at T=2.0, got {report.fallback_slices}"
    )
    assert report.failed_slices == []
    assert report.repair_infeasible is True, (
        "repair_infeasible must be True when a fallback slice violates H&M"
    )
    assert report.metrics.n_violations_after >= 1, (
        "the remaining calendar violation must be surfaced, not hidden"
    )


def test_repair_essvi_routes_degenerate_corner_to_fallback() -> None:
    """The m66 post-fit margin check must route degenerate H&M boundary
    corners to the fallback path (docs/code_review_findings.md §6.7).

    Same dip fixture as the xfail tripwire above.  With the fix, BOTH
    theta-dipping slices converge to degenerate corners pinned at the
    H&M eps floors (theta_delta=eps_theta, chi_delta=eps_chi,
    ratio~1.0) with anomalously bad per-slice RMSE:

    - T=0.5 flattened onto T=0.25 (hard RMSE ~0.0519 vs the
      unconstrained fit's ~1.3e-11), and
    - T=2.0 flattened onto T=1.0 (hard RMSE ~0.0499 vs ~1.1e-13).

    The margin check routes BOTH to fallback_slices; the unconstrained
    per-slice fallback then recovers the true theta dips (0.03 at
    T=0.5, 0.07 at T=2.0), and the report honestly flags the remaining
    calendar violations (repair_infeasible=True,
    n_violations_after >= 1).  Pre-fix, both corners were silently
    certified arb-free (fallback_slices=[], repair_infeasible=False).
    """
    report = repair(_ssvi_priced_surface(_DIP_TRUTH_ENGINE), use_ssvi=True)

    assert 2.0 in report.fallback_slices, (
        f"expected T=2.0 in fallback_slices, got {report.fallback_slices}"
    )
    assert 0.5 in report.fallback_slices, (
        f"expected T=0.5 in fallback_slices (also a degenerate corner), "
        f"got {report.fallback_slices}"
    )
    assert report.failed_slices == [], (
        f"expected no failed slices, got {report.failed_slices}"
    )
    # Every slice still gets a fit (hard or fallback).
    assert report.metrics.n_slices_fitted == 4, (
        f"expected 4 fitted slices, got {report.metrics.n_slices_fitted}"
    )
    assert len(report.fitted_ssvi_slices) == 4
    assert report.repair_infeasible is True, (
        "repair_infeasible must be True when corners route to fallback"
    )
    assert report.metrics.n_violations_after >= 1, (
        "the remaining calendar violation must be surfaced, not hidden"
    )


def test_repair_essvi_skips_slice_with_few_points() -> None:
    """A slice with fewer than 5 (k,w) points is skipped from the eSSVI
    fit with a warning — not fitted, not failed, no crash."""
    truth = _DIP_TRUTH_ENGINE + [(3.0, dict(theta=0.10, rho=0.1, psi=0.5))]
    surface = _ssvi_priced_surface(truth)
    # Shrink the T=3.0 slice to 2 strikes (4 quotes -> 4 (k,w) points)
    small = surface.slices[-1]
    surface.slices[-1] = ExpirySlice(
        expiry_time=small.expiry_time,
        quotes=small.quotes[:4],
    )

    report = repair(surface, use_ssvi=True)

    assert report.metrics.n_slices_input == 5
    assert report.metrics.n_slices_fitted == 4, (
        f"expected the tiny slice to be skipped, got "
        f"{report.metrics.n_slices_fitted} fitted"
    )
    fitted_Ts = [s.expiry_time for s in report.fitted_ssvi_slices]
    assert 3.0 not in fitted_Ts
    assert 3.0 not in report.failed_slices
    assert 3.0 not in report.fallback_slices


def test_repair_essvi_reports_slice_with_no_fit(monkeypatch) -> None:
    """When both the hard fit and the unconstrained fallback fail for a
    slice, repair() records it in failed_slices and skips it from the
    fitted output with a warning — no crash, no silent hole."""
    import arbfree_vol.ssvi.term_structure as ts

    def _always_raise(points, prev=None, **kwargs):
        raise RuntimeError("simulated total fit failure")

    def _no_fallback(points):
        raise RuntimeError("simulated fallback failure")

    monkeypatch.setattr(ts, "_fit_slice", _always_raise)
    monkeypatch.setattr(ts, "fit_ssvi_slice", _no_fallback)

    report = repair(_ssvi_priced_surface(_DIP_TRUTH_ENGINE), use_ssvi=True)

    assert report.failed_slices == [0.25, 0.5, 1.0, 2.0], (
        f"expected all expiries in failed_slices, got {report.failed_slices}"
    )
    assert report.metrics.n_slices_fitted == 0
    assert len(report.fitted_ssvi_slices) == 0


# ── Raw-SVI slice bookkeeping (mirrors the eSSVI path above) ────────────
# The default (raw SVI) repair path must treat a slice whose constrained
# calibration fails entirely the same way the eSSVI path treats a total
# fit failure: recorded in failed_slices, excluded from the fitted output,
# and logged — never silently dropped.


def test_repair_svi_reports_slice_with_no_fit(monkeypatch) -> None:
    """When the raw-SVI constrained calibration fails for a slice,
    repair() records it in failed_slices and skips it from the fitted
    output with a warning — no crash, no silent hole.

    Mirrors ``test_repair_essvi_reports_slice_with_no_fit`` for the
    default (raw SVI) path.
    """
    import arbfree_vol.repair.engine as engine_mod

    def _raise(*args, **kwargs):
        raise RuntimeError("simulated calibration failure")

    monkeypatch.setattr(engine_mod, "calibrate_constrained", _raise)

    report = repair(_svi_priced_surface(_SVI_TRUTH_ENGINE))

    assert report.failed_slices == [0.25, 1.0], (
        f"expected both expiries in failed_slices, got {report.failed_slices}"
    )
    assert report.metrics.n_slices_fitted == 0
    assert len(report.fitted_slices) == 0


def test_repair_svi_skips_slice_with_few_points() -> None:
    """A slice with fewer than 5 (k,w) points is skipped from the raw-SVI
    fit with a warning — not fitted, not failed, not a fallback.  This is
    a SKIP (like the eSSVI path), not a failure, so the expiry must NOT
    appear in failed_slices."""
    surface = _svi_priced_surface(_SVI_TRUTH_ENGINE)
    # Shrink the T=1.0 slice to 4 quotes (2 strikes -> 2 (k,w) points)
    small = surface.slices[-1]
    surface.slices[-1] = ExpirySlice(
        expiry_time=small.expiry_time,
        quotes=small.quotes[:4],
    )

    report = repair(surface)

    assert report.metrics.n_slices_input == 2
    assert report.metrics.n_slices_fitted == 1, (
        f"expected the tiny slice to be skipped, got "
        f"{report.metrics.n_slices_fitted} fitted"
    )
    fitted_Ts = [s.expiry_time for s in report.fitted_slices]
    assert 1.0 not in fitted_Ts
    assert 1.0 not in report.failed_slices
    assert 1.0 not in report.fallback_slices


def test_sabr_failure_marks_failed_slices(monkeypatch) -> None:
    """When the SABR term-structure fit raises RuntimeError, repair()
    must not crash: it marks every SABR-eligible expiry as failed,
    returns no SABR fits, and surfaces the failure honestly.

    Previously the fabricated-default fallback in
    ``fit_sabr_term_structure`` silently returned SABRParams(0.2, 0.5,
    0.0, 0.3) that flowed into fitted_sabr_slices as if fitted.
    """
    import arbfree_vol.repair.engine as engine_mod

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(engine_mod, "fit_sabr_term_structure", _raise)

    expiries = [0.25, 0.5, 1.0]
    n_strikes = 7
    strikes = [SPOT * (1 + 0.1 * (i - n_strikes // 2)) for i in range(n_strikes)]

    slices: list[ExpirySlice] = []
    for T_val in expiries:
        quotes: list[Quote] = []
        for K in strikes:
            quotes.append(
                Quote(strike=K, option_type=OptionType.CALL,
                      price=_bs_price(OptionType.CALL, K, sigma=0.2, tt=T_val))
            )
            quotes.append(
                Quote(strike=K, option_type=OptionType.PUT,
                      price=_bs_price(OptionType.PUT, K, sigma=0.2, tt=T_val))
            )
        slices.append(ExpirySlice(expiry_time=T_val, quotes=quotes))

    surface = VolSurface(spot=SPOT, risk_free=R, div_yield=Q, slices=slices)

    # Must not raise
    report = repair(surface, use_sabr=True)

    assert report.failed_slices == expiries, (
        f"expected all SABR-eligible expiries in failed_slices, got "
        f"{report.failed_slices}"
    )
    assert report.fitted_sabr_slices == ()
    assert len(report.fitted_slices) == 0


def test_repair_infeasible_true_when_grid_finds_remaining_violations() -> None:
    """repair_infeasible must be True when the grid-based
    detect_svi_surface finds remaining violations, even if the eSSVI
    H&M parameter check passed.

    The 7-strike variant of the dip ground truth fits all slices hard
    (no fallback), so verify_hm_condition passes and the pre-fix code
    kept repair_infeasible=False — yet the raw-SVI grid detects 2
    calendar violations on the fitted surface.
    """
    report = repair(
        _ssvi_priced_surface(_DIP_TRUTH_ENGINE, n_strikes=7), use_ssvi=True,
    )

    assert report.fallback_slices == [], (
        f"test setup error: expected no fallback, got {report.fallback_slices}"
    )
    assert report.metrics.n_violations_after >= 1, (
        "test setup error: expected remaining grid violations on the "
        "7-strike dip surface"
    )
    assert report.repair_infeasible is True, (
        "repair_infeasible must be True when the grid detects remaining "
        "violations"
    )


def test_repair_infeasible_true_for_wing_crossing_beyond_svi_grid() -> None:
    """repair_infeasible must be True when the fitted eSSVI slices cross
    in the wings OUTSIDE the [-1.5, 1.5] raw-SVI detector grid.

    The crossing pair passes verify_hm_condition AND the raw-SVI
    detect_svi_surface grid (the crossing sits at k ~ -1.73, beyond the
    detector grid).  Only the native-calendar gate in the engine catches
    it.  fallback_slices and failed_slices must stay empty — the fit
    succeeded; it is the certification that must fail.
    """
    import arbfree_vol.repair.engine as engine_mod
    from arbfree_vol.ssvi.model import SSVIParams
    from arbfree_vol.ssvi.term_structure import SequentialFitResult

    crossing_params = [
        (0.25, SSVIParams(theta=0.5, rho=0.00, psi=1.0)),
        (1.00, SSVIParams(theta=1.0, rho=0.58, psi=1.2)),
    ]
    fake_result = SequentialFitResult(
        fitted_slices=crossing_params,
        fallback_slices=[],
        failed_slices=[],
    )

    def _fake_sequential(slices_data):
        return fake_result

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(engine_mod, "fit_ssvi_surface_sequential", _fake_sequential)

    try:
        report = repair(
            _ssvi_priced_surface(_DIP_TRUTH_ENGINE, n_strikes=9), use_ssvi=True,
        )
    finally:
        monkeypatch.undo()

    # The fake fit reports success: no fallback and no failed slices.
    assert report.fallback_slices == []
    assert report.failed_slices == []
    # The wing crossing is NOT visible to the raw-SVI detector grid, yet
    # the native-calendar gate must flag the surface as infeasible.
    assert report.repair_infeasible is True, (
        "repair_infeasible must be True when fitted eSSVI slices cross "
        "outside the raw-SVI detector grid"
    )


def test_sabr_mapping_success_records_slices_in_fitted_outputs(monkeypatch) -> None:
    """When the SABR->SVI mapping succeeds, every slice must appear in
    the fitted outputs and none may be recorded as failed-mapping.

    Deterministic by construction: the SABR term-structure fit and the
    SABR->SVI mapping are both monkeypatched with cheap stubs, so the
    outcome cannot depend on optimizer budgets.  This is the SUCCESS
    counterpart to the failure-accounting tests below (RuntimeError /
    ValueError mapping raises), which together replace the old
    all-succeeded-or-all-failed conditional pair.
    """
    import arbfree_vol.repair.engine as engine_mod

    expiries = [0.25, 0.5, 1.0]
    params = [SABRParams(alpha=0.1, beta=0.5, rho=0.0, nu=0.5)
              for _ in expiries]

    def _cheap_mapping(sabr_params, forward_price, expiry_time):
        # Valid raw-SVI tuple (a, b, rho, m, sigma).
        return (0.04, 0.4, -0.4, 0.0, 0.1)

    monkeypatch.setattr(engine_mod, "fit_sabr_term_structure",
                        lambda slices_data: params)
    monkeypatch.setattr(engine_mod, "sabr_to_raw_svi_params", _cheap_mapping)

    report = repair(_flat_bs_surface(expiries), use_sabr=True)

    # Every slice accounted for as fitted; no mapping failures.
    assert report.sabr_mapping_failed_slices == [], (
        f"expected no failed-mapping slices, got "
        f"{report.sabr_mapping_failed_slices}"
    )
    assert len(report.fitted_sabr_slices) == len(expiries), (
        f"expected all {len(expiries)} slices in fitted_sabr_slices, got "
        f"{len(report.fitted_sabr_slices)}"
    )
    assert len(report.fitted_slices) == len(expiries), (
        f"expected all {len(expiries)} slices in fitted_slices, got "
        f"{len(report.fitted_slices)}"
    )
    fitted_Ts = sorted(fs.expiry_time for fs in report.fitted_sabr_slices)
    assert fitted_Ts == expiries


@pytest.mark.slow
# Slow: runs the REAL SABR->SVI mapping (sabr_to_raw_svi_params), which
# burns thousands of optimizer evals per slice (the adversarial combo
# needs ~10k nfev per slice; ~30-60s total).
def test_sabr_mapping_real_optimizer_accounts_for_every_slice(monkeypatch) -> None:
    """Exercise the REAL SABR->SVI optimizer on a difficult parameter
    regime and assert the deterministic accounting contract: no slice may
    be silently dropped.

    The SABR term-structure fit is stubbed (fast), but
    ``sabr_to_raw_svi_params`` is the REAL optimizer on an adversarial
    combo (alpha=3.0, rho=0.995, nu=0.2 — a near-boundary rho with a
    large alpha).  Whatever the optimizer does — converge within its
    50,000-eval budget, or fail and raise — the repair path must account
    for every slice: each expiry is either present in
    ``fitted_sabr_slices`` or recorded in ``sabr_mapping_failed_slices``,
    never both, never neither, and the raw-SVI ``fitted_slices`` mirror
    the SABR fitted output 1:1.

    This is the deterministic contract restored from
    ``test_sabr_mapping_adversarial_params_no_crash`` (deleted in
    ce4f4d8): unlike the old test it does NOT branch its assertions on
    the optimizer outcome — the accounting invariant must hold no matter
    which branch the optimizer takes.
    """
    import arbfree_vol.repair.engine as engine_mod

    expiries = [0.25, 0.5, 1.0]
    adversarial = [SABRParams(alpha=3.0, beta=0.5, rho=0.995, nu=0.2)
                   for _ in expiries]

    monkeypatch.setattr(engine_mod, "fit_sabr_term_structure",
                        lambda slices_data: adversarial)

    report = repair(_flat_bs_surface(expiries), use_sabr=True)

    # Deterministic contract: every slice is accounted for — fitted or
    # failed-mapping — never silently dropped.
    fitted_Ts = sorted(fs.expiry_time for fs in report.fitted_sabr_slices)
    failed_Ts = sorted(report.sabr_mapping_failed_slices)
    assert sorted(fitted_Ts + failed_Ts) == expiries, (
        f"expected every expiry accounted for (fitted or failed-mapping), "
        f"got fitted={fitted_Ts}, failed={failed_Ts}"
    )
    # A slice cannot be both fitted and failed-mapping.
    assert set(fitted_Ts).isdisjoint(failed_Ts), (
        f"slice recorded as both fitted and failed-mapping: "
        f"fitted={fitted_Ts}, failed={failed_Ts}"
    )
    # The raw-SVI fitted_slices mirror the SABR fitted output 1:1 (each
    # successfully mapped SABR slice becomes exactly one FittedSlice).
    assert len(report.fitted_slices) == len(report.fitted_sabr_slices), (
        f"expected fitted_slices to mirror fitted_sabr_slices, got "
        f"{len(report.fitted_slices)} vs {len(report.fitted_sabr_slices)}"
    )
    assert report.metrics.n_slices_fitted == len(report.fitted_sabr_slices), (
        f"metrics must count the fitted SABR slices, got "
        f"{report.metrics.n_slices_fitted}"
    )


def test_sabr_mapping_wrap_records_failed_slices(caplog, monkeypatch) -> None:
    """The wrap around ``sabr_to_raw_svi_params`` (option A) is the
    correctness guarantee: when a slice's mapping raises RuntimeError,
    repair() must not crash — the slice is recorded in
    ``sabr_mapping_failed_slices``, excluded from ``fitted_sabr_slices``
    and ``fitted_slices``, and a WARNING is logged.  No max_nfev is
    provably sufficient over a continuous parameter space, so this is the
    path that keeps repair() usable when some future real slice exceeds
    even the raised budget.
    """
    import arbfree_vol.repair.engine as engine_mod

    expiries = [0.25, 0.5, 1.0]

    def _raise_mapping(*args, **kwargs):
        raise RuntimeError("mapping boom")

    monkeypatch.setattr(engine_mod, "sabr_to_raw_svi_params", _raise_mapping)

    with caplog.at_level(logging.WARNING, logger="arbfree_vol.repair.engine"):
        report = repair(_flat_bs_surface(expiries), use_sabr=True)

    # No exception propagates; every slice is recorded as failed-mapping.
    assert sorted(report.sabr_mapping_failed_slices) == expiries, (
        f"expected all expiries in sabr_mapping_failed_slices, got "
        f"{report.sabr_mapping_failed_slices}"
    )
    # The mapping failure excludes the slices from both fitted outputs.
    assert report.fitted_sabr_slices == ()
    assert len(report.fitted_slices) == 0
    # The failure is logged, not swallowed silently.
    assert "mapping boom" in caplog.text
    assert "SABR->SVI mapping failed for slice" in caplog.text


def test_sabr_mapping_wrap_records_failed_slices_on_value_error(caplog, monkeypatch) -> None:
    """The wrap around ``sabr_to_raw_svi_params`` must also catch the
    ValueError that scipy's ``least_squares`` raises on non-finite
    residuals.

    Pre-fix, only RuntimeError was caught: a ValueError escaped the wrap
    and aborted ``repair()`` entirely.  Post-fix the slice is logged and
    recorded in ``sabr_mapping_failed_slices`` — same honest bookkeeping
    as the RuntimeError path.
    """
    import arbfree_vol.repair.engine as engine_mod

    expiries = [0.25, 0.5, 1.0]

    def _raise_value_error(*args, **kwargs):
        raise ValueError("array must not contain infs or NaNs")

    monkeypatch.setattr(engine_mod, "sabr_to_raw_svi_params", _raise_value_error)

    with caplog.at_level(logging.WARNING, logger="arbfree_vol.repair.engine"):
        report = repair(_flat_bs_surface(expiries), use_sabr=True)

    # No exception propagates; every slice is recorded as failed-mapping.
    assert sorted(report.sabr_mapping_failed_slices) == expiries, (
        f"expected all expiries in sabr_mapping_failed_slices, got "
        f"{report.sabr_mapping_failed_slices}"
    )
    # The mapping failure excludes the slices from both fitted outputs.
    assert report.fitted_sabr_slices == ()
    assert len(report.fitted_slices) == 0
    # The failure is logged distinctly (mentions ValueError), not swallowed.
    assert "array must not contain infs or NaNs" in caplog.text
    assert "SABR->SVI mapping failed for slice" in caplog.text
    assert "ValueError" in caplog.text


# ── No-forward-estimate slice bookkeeping (all three repair paths) ──────
# A slice whose forward price is missing from the estimated forward curve
# (``fwd_curve.get(expiry) is None``) cannot be fitted at all.  The three
# repair paths used to silently ``continue`` past such slices; they now log
# a warning and record the expiry in ``failed_slices`` — same "report,
# don't raise" philosophy as the calibration-failure bookkeeping.


def test_repair_svi_records_slice_with_no_forward(caplog, monkeypatch) -> None:
    """When the forward curve has no estimate for a slice, the raw-SVI
    path must record the slice in failed_slices (with a warning), fit the
    remaining slices, and not crash."""
    import arbfree_vol.repair.engine as engine_mod

    monkeypatch.setattr(
        engine_mod, "estimate_forward_curve", _forward_curve_missing(1.0),
    )

    with caplog.at_level(logging.WARNING, logger="arbfree_vol.repair.engine"):
        report = repair(_svi_priced_surface(_SVI_TRUTH_ENGINE))

    assert report.failed_slices == [1.0], (
        f"expected the no-forward expiry in failed_slices, got "
        f"{report.failed_slices}"
    )
    assert report.metrics.n_slices_fitted == 1, (
        f"expected the 0.25 slice to be fitted, got "
        f"{report.metrics.n_slices_fitted}"
    )
    assert [s.expiry_time for s in report.fitted_slices] == [0.25]
    assert "SVI path: no forward estimate for slice T=1.0000" in caplog.text


def test_repair_essvi_records_slice_with_no_forward(caplog, monkeypatch) -> None:
    """The eSSVI path must record a no-forward slice in failed_slices
    even though ``failed_slices`` is REASSIGNED from the sequential fit
    result after the loop — the no-forward expiries are collected first
    and re-appended after the reassignment (the collect-then-extend
    logic)."""
    import arbfree_vol.repair.engine as engine_mod

    truth = [
        (0.25, dict(theta=0.08, rho=-0.3, psi=0.5)),
        (0.50, dict(theta=0.12, rho=-0.2, psi=0.5)),
    ]

    monkeypatch.setattr(
        engine_mod, "estimate_forward_curve", _forward_curve_missing(0.5),
    )

    with caplog.at_level(logging.WARNING, logger="arbfree_vol.repair.engine"):
        report = repair(_ssvi_priced_surface(truth), use_ssvi=True)

    assert report.failed_slices == [0.5], (
        f"expected the no-forward expiry in failed_slices, got "
        f"{report.failed_slices}"
    )
    assert report.metrics.n_slices_fitted == 1, (
        f"expected the 0.25 slice to be fitted, got "
        f"{report.metrics.n_slices_fitted}"
    )
    assert [s.expiry_time for s in report.fitted_ssvi_slices] == [0.25]
    assert "eSSVI path: no forward estimate for slice T=0.5000" in caplog.text


def test_repair_essvi_failed_slices_sorted_with_no_forward(monkeypatch) -> None:
    """``RepairReport.failed_slices`` must be SORTED by expiry and
    deduplicated even when the no-forward expiry sorts BEFORE a
    failed-fit expiry.

    The eSSVI path reassigns ``failed_slices`` from the sequential fit
    result (``[0.5]`` here: the 0.5 slice fails both fits) and then
    appends the no-forward expiries (``[0.25]``: no forward estimate),
    which produces ``[0.5, 0.25]`` without the ordering contract.  The
    documented contract (``repair`` docstring) is chronological order,
    so the report must carry ``[0.25, 0.5]``."""
    import arbfree_vol.ssvi.term_structure as ts
    import arbfree_vol.repair.engine as engine_mod

    truth = [
        (0.25, dict(theta=0.08, rho=-0.3, psi=0.5)),
        (0.50, dict(theta=0.12, rho=-0.2, psi=0.5)),
    ]
    surface = _ssvi_priced_surface(truth)

    # T=0.25 has no forward estimate; T=0.5 fails both the hard and the
    # unconstrained fallback fits.
    monkeypatch.setattr(
        engine_mod, "estimate_forward_curve", _forward_curve_missing(0.25),
    )

    def _always_raise(points, prev=None, **kwargs):
        raise RuntimeError("simulated total fit failure")

    def _no_fallback(points):
        raise RuntimeError("simulated fallback failure")

    monkeypatch.setattr(ts, "_fit_slice", _always_raise)
    monkeypatch.setattr(ts, "fit_ssvi_slice", _no_fallback)

    report = repair(surface, use_ssvi=True)

    assert report.failed_slices == [0.25, 0.5], (
        f"failed_slices must be sorted by expiry, got "
        f"{report.failed_slices}"
    )
    assert report.metrics.n_slices_fitted == 0
    assert len(report.fitted_ssvi_slices) == 0


def test_repair_sabr_records_slice_with_no_forward(caplog, monkeypatch) -> None:
    """The SABR path must record a no-forward slice in failed_slices
    (this path never reassigns ``failed_slices``, so the append is
    direct), fit the remaining slices, and not crash."""
    import arbfree_vol.repair.engine as engine_mod

    monkeypatch.setattr(
        engine_mod, "estimate_forward_curve", _forward_curve_missing(0.5),
    )

    with caplog.at_level(logging.WARNING, logger="arbfree_vol.repair.engine"):
        report = repair(_flat_bs_surface([0.25, 0.5]), use_sabr=True)

    assert report.failed_slices == [0.5], (
        f"expected the no-forward expiry in failed_slices, got "
        f"{report.failed_slices}"
    )
    assert report.metrics.n_slices_fitted == 1, (
        f"expected the 0.25 slice to be fitted, got "
        f"{report.metrics.n_slices_fitted}"
    )
    assert [s.expiry_time for s in report.fitted_sabr_slices] == [0.25]
    assert "SABR path: no forward estimate for slice T=0.5000" in caplog.text
