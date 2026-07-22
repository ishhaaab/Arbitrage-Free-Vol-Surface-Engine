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
