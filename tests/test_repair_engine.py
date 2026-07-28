"""Tests for the repair engine."""
from datetime import date
import pytest
from pytest import approx

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.repair.engine import repair


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
