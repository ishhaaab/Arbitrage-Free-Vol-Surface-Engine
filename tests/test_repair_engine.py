"""Tests for the repair engine."""
from datetime import date
import logging
import pytest
from pytest import approx

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.repair.engine import repair
from arbfree_vol.sabr.model import SABRParams


SPOT = 100.0
R = 0.05
Q = 0.0
T = 1.0
_DUMMY_DATE = date(2030, 1, 1)


def _bs_price(otype: OptionType, strike: float,
              sigma: float = 0.2, tt: float = T) -> float:
    from arbfree_vol.models.option import OptionContract, BlackScholesInput
    from arbfree_vol.pricing.black_scholes import price

    contract = OptionContract(
        symbol="X", option_type=otype, strike=strike,
        expiry_date=_DUMMY_DATE,
    )
    model = BlackScholesInput(
        contract=contract, spot=SPOT, expiry_time=tt,
        risk_free=R, div_yield=Q, volatility=sigma,
    )
    return price(model)


def _clean_surface(n_strikes: int = 7) -> VolSurface:
    """Build a surface with calls and puts across n_strikes, all priced at
    sigma=0.2 from the same model — no arb violations by construction."""
    strikes = [SPOT * (1 + 0.1 * (i - n_strikes // 2)) for i in range(n_strikes)]
    quotes: list[Quote] = []
    for K in strikes:
        quotes.append(
            Quote(strike=K, option_type=OptionType.CALL,
                  price=_bs_price(OptionType.CALL, K))
        )
        quotes.append(
            Quote(strike=K, option_type=OptionType.PUT,
                  price=_bs_price(OptionType.PUT, K))
        )

    return VolSurface(
        spot=SPOT, risk_free=R, div_yield=Q,
        slices=[ExpirySlice(expiry_time=T, quotes=quotes)],
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


# ── Real (non-monkeypatched) fallback through the engine ─────────────
# Same genuinely H&M-incompatible ground truth as the term-structure
# tests, but priced into a VolSurface so the FULL repair() pipeline
# runs: cleaning -> forward curve -> sequential eSSVI fit -> engine
# fallback bookkeeping.  The T=2.0 slice really falls back (the
# live-data linesearch failure), repair_infeasible goes True, and the
# remaining violation is surfaced — nothing is silently certified.
_DIP_TRUTH_ENGINE = [
    (0.25, dict(theta=0.08, rho=-0.3, psi=0.5)),
    (0.50, dict(theta=0.03, rho=-0.2, psi=0.35)),  # theta + chi dip
    (1.00, dict(theta=0.12, rho=0.1, psi=0.4)),
    (2.00, dict(theta=0.07, rho=0.2, psi=0.55)),   # theta dip again
]


def _ssvi_priced_surface(truth, n_strikes: int | None = None) -> VolSurface:
    """Price a surface from SSVI ground truth so the (k, w) data the
    engine sees matches the fitted model's conventions exactly."""
    from math import sqrt, exp
    from arbfree_vol.ssvi.model import ssvi_w

    ks = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    if n_strikes is not None:
        ks = ks[:n_strikes]
    slices: list[ExpirySlice] = []
    for T, t in truth:
        F = SPOT * exp((R - Q) * T)
        quotes: list[Quote] = []
        for k in ks:
            K = F * exp(k)
            w = ssvi_w(k, t["theta"], t["rho"], t["psi"])
            sigma = sqrt(max(w / T, 1e-12))
            quotes.append(Quote(
                strike=K, option_type=OptionType.CALL,
                price=_bs_price(OptionType.CALL, K, sigma=sigma, tt=T),
            ))
            quotes.append(Quote(
                strike=K, option_type=OptionType.PUT,
                price=_bs_price(OptionType.PUT, K, sigma=sigma, tt=T),
            ))
        slices.append(ExpirySlice(expiry_time=T, quotes=quotes))
    return VolSurface(spot=SPOT, risk_free=R, div_yield=Q, slices=slices)


def test_repair_essvi_real_fallback_on_incompatible_data() -> None:
    """End-to-end: genuinely H&M-incompatible data through repair()
    produces a REAL fallback (no monkeypatch) that is honestly flagged.

    The T=2.0 slice falls back (the live-data linesearch failure),
    repair_infeasible is True, and the remaining calendar violation is
    surfaced in n_violations_after — the surface is never silently
    certified arb-free.  Deleting the hard constraints in _fit_slice
    would let the unconstrained fit recover the dips with no fallback
    recorded, and this test would fail.
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


def _flat_bs_surface(expiries: list[float], n_strikes: int = 7) -> VolSurface:
    """Build a clean multi-slice surface priced at flat BS vol 0.2.

    Mirrors the surface construction used by the existing SABR repair
    tests (``test_sabr_failure_marks_failed_slices`` etc.) so the
    mapping-failure tests exercise the same pipeline path.
    """
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

    return VolSurface(spot=SPOT, risk_free=R, div_yield=Q, slices=slices)


@pytest.mark.slow
# Slow: runs the real SABR->SVI mapping (sabr_to_raw_svi_params), which
# burns thousands of optimizer evals per slice (~65s).
def test_sabr_mapping_adversarial_params_no_crash(monkeypatch) -> None:
    """The adversarial SABR combo (alpha=3.0, rho=0.995, nu=0.2) must not
    crash repair(): at max_nfev=50000 the mapping converges (adversarial
    scan: nfev ~10389) and every slice lands in fitted_sabr_slices.

    The primary contract is "never crashes"; the fitted/failed accounting
    is asserted robustly so the test survives future budget changes.
    """
    import arbfree_vol.repair.engine as engine_mod

    expiries = [0.25, 0.5, 1.0]
    adversarial = [SABRParams(alpha=3.0, beta=0.5, rho=0.995, nu=0.2)
                   for _ in expiries]

    monkeypatch.setattr(engine_mod, "fit_sabr_term_structure",
                        lambda slices_data: adversarial)

    report = repair(_flat_bs_surface(expiries), use_sabr=True)

    # Primary contract: no exception propagates out of repair().
    # Every slice must be accounted for — fitted or failed-mapping.
    assert len(report.fitted_sabr_slices) + len(report.sabr_mapping_failed_slices) == len(expiries), (
        f"expected {len(expiries)} slices accounted for, got fitted="
        f"{len(report.fitted_sabr_slices)}, failed={report.sabr_mapping_failed_slices}"
    )
    if report.fitted_sabr_slices:
        # Strong expectation: the adversarial scan converges at
        # nfev=10389 (< 50000), so every slice should be fitted and
        # nothing should be recorded as failed-mapping.
        assert len(report.fitted_sabr_slices) == len(expiries), (
            f"expected all slices fitted, got {len(report.fitted_sabr_slices)}"
        )
        assert report.sabr_mapping_failed_slices == [], (
            f"expected no failed-mapping slices, got {report.sabr_mapping_failed_slices}"
        )
    else:
        # Robustness guard: if the mapping budget is ever lowered below
        # what this combo needs, every slice must be recorded as
        # failed-mapping rather than silently dropped.
        assert len(report.sabr_mapping_failed_slices) == len(expiries), (
            f"expected all expiries in sabr_mapping_failed_slices, got "
            f"{report.sabr_mapping_failed_slices}"
        )


@pytest.mark.slow
# Slow: runs the real SABR->SVI mapping (sabr_to_raw_svi_params), which
# burns thousands of optimizer evals per slice (~55s).
def test_sabr_mapping_real_data_params_no_crash(monkeypatch) -> None:
    """The real-data SABR combo (alpha=0.00243, rho=-0.484, nu=2.72) must
    not crash repair(): it needs 13,685 evaluations — the documented case
    that used to crash at the old 10,000 budget — and now succeeds at
    max_nfev=50000.

    Same primary contract as the adversarial test: never crash, and keep
    the fitted/failed accounting robust.
    """
    import arbfree_vol.repair.engine as engine_mod

    expiries = [0.25, 0.5, 1.0]
    real_data = [SABRParams(alpha=0.00243, beta=0.5, rho=-0.484, nu=2.72)
                 for _ in expiries]

    monkeypatch.setattr(engine_mod, "fit_sabr_term_structure",
                        lambda slices_data: real_data)

    report = repair(_flat_bs_surface(expiries), use_sabr=True)

    assert len(report.fitted_sabr_slices) + len(report.sabr_mapping_failed_slices) == len(expiries), (
        f"expected {len(expiries)} slices accounted for, got fitted="
        f"{len(report.fitted_sabr_slices)}, failed={report.sabr_mapping_failed_slices}"
    )
    if report.fitted_sabr_slices:
        # Strong expectation: this is the documented 13,685-eval case
        # that used to crash at max_nfev=10000; at 50000 it fits.
        assert len(report.fitted_sabr_slices) == len(expiries), (
            f"expected all slices fitted, got {len(report.fitted_sabr_slices)}"
        )
        assert report.sabr_mapping_failed_slices == []
    else:
        assert len(report.sabr_mapping_failed_slices) == len(expiries), (
            f"expected all expiries in sabr_mapping_failed_slices, got "
            f"{report.sabr_mapping_failed_slices}"
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
