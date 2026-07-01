"""Tests for static arbitrage detection."""

from datetime import date

from pytest import approx

from arbfree_vol.arbitrage.quote_detect import detect
from arbfree_vol.arbitrage.report import ViolationType
from arbfree_vol.models.option import (
    BlackScholesInput,
    OptionContract,
    OptionType,
)
from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface
from arbfree_vol.pricing.black_scholes import price

SPOT = 100.0
RISK_FREE = 0.05
DIV_YIELD = 0.0
T = 1.0


def _bs_price(option_type: OptionType, strike: float, sigma: float = 0.2) -> float:
    contract = OptionContract(
        symbol="NVDA",
        option_type=option_type,
        strike=strike,
        expiry_date=date(2026, 11, 27),
    )
    model = BlackScholesInput(
        contract=contract,
        spot=SPOT,
        expiry_time=T,
        risk_free=RISK_FREE,
        div_yield=DIV_YIELD,
        volatility=sigma,
    )
    return price(model)


def _surface(quotes: list[Quote]) -> VolSurface:
    return VolSurface(
        spot=SPOT,
        risk_free=RISK_FREE,
        div_yield=DIV_YIELD,
        slices=[ExpirySlice(expiry_time=T, quotes=quotes)],
    )


def test_parity_consistent_surface_is_arbitrage_free() -> None:
    # Call and put generated from the same model => parity holds by construction.
    call = _bs_price(OptionType.CALL, 100.0)
    put = _bs_price(OptionType.PUT, 100.0)
    surface = _surface(
        [
            Quote(strike=100.0, option_type=OptionType.CALL, price=call),
            Quote(strike=100.0, option_type=OptionType.PUT, price=put),
        ]
    )

    report = detect(surface)

    assert report.is_arbitrage_free
    assert report.violations == []


def test_parity_violation_is_detected() -> None:
    call = _bs_price(OptionType.CALL, 100.0)
    put = _bs_price(OptionType.PUT, 100.0)
    surface = _surface(
        [
            Quote(strike=100.0, option_type=OptionType.CALL, price=call),
            # bump the put by 1.0 -> breaks C - P by exactly 1.0
            Quote(strike=100.0, option_type=OptionType.PUT, price=put + 1.0),
        ]
    )

    report = detect(surface)

    assert not report.is_arbitrage_free
    assert len(report.violations) == 1
    v = report.violations[0]
    assert v.kind == ViolationType.PARITY
    assert v.magnitude == approx(1.0, abs=1e-6)


def test_monotonicity_violation_is_detected() -> None:
    # Call prices must fall as strike rises; here the 110 call is dearer than the
    # 100 call -> a vertical-spread arbitrage.
    surface = _surface(
        [
            Quote(strike=100.0, option_type=OptionType.CALL, price=5.0),
            Quote(strike=110.0, option_type=OptionType.CALL, price=6.0),
        ]
    )

    report = detect(surface)

    assert not report.is_arbitrage_free
    kinds = [v.kind for v in report.violations]
    assert ViolationType.MONOTONICITY in kinds
    mono = next(v for v in report.violations if v.kind == ViolationType.MONOTONICITY)
    assert mono.magnitude == approx(1.0, abs=1e-6)


def test_monotonic_calls_are_arbitrage_free() -> None:
    # Properly decreasing call prices across strikes -> no monotonicity violation.
    surface = _surface(
        [
            Quote(strike=90.0, option_type=OptionType.CALL, price=_bs_price(OptionType.CALL, 90.0)),
            Quote(strike=100.0, option_type=OptionType.CALL, price=_bs_price(OptionType.CALL, 100.0)),
            Quote(strike=110.0, option_type=OptionType.CALL, price=_bs_price(OptionType.CALL, 110.0)),
        ]
    )

    report = detect(surface)

    assert report.is_arbitrage_free


def test_butterfly_violation_is_detected() -> None:
    # Strikes 90/100/110 (evenly spaced -> line = average of outer two = 6.0).
    # A middle call of 8.0 pokes 2.0 above the line -> negative density.
    surface = _surface(
        [
            Quote(strike=90.0, option_type=OptionType.CALL, price=10.0),
            Quote(strike=100.0, option_type=OptionType.CALL, price=8.0),
            Quote(strike=110.0, option_type=OptionType.CALL, price=2.0),
        ]
    )

    report = detect(surface)

    assert not report.is_arbitrage_free
    fly = next(v for v in report.violations if v.kind == ViolationType.BUTTERFLY)
    assert fly.magnitude == approx(2.0, abs=1e-6)


def test_convex_calls_are_arbitrage_free() -> None:
    # Call prices from a flat-vol model are convex in strike by construction.
    surface = _surface(
        [
            Quote(strike=90.0, option_type=OptionType.CALL, price=_bs_price(OptionType.CALL, 90.0)),
            Quote(strike=100.0, option_type=OptionType.CALL, price=_bs_price(OptionType.CALL, 100.0)),
            Quote(strike=110.0, option_type=OptionType.CALL, price=_bs_price(OptionType.CALL, 110.0)),
        ]
    )

    report = detect(surface)

    assert report.is_arbitrage_free


def _call_quote(strike: float, sigma: float, t: float) -> Quote:
    """A call quote priced at a given vol and maturity."""
    contract = OptionContract(
        symbol="NVDA",
        option_type=OptionType.CALL,
        strike=strike,
        expiry_date=date(2026, 11, 27),
    )
    model = BlackScholesInput(
        contract=contract,
        spot=SPOT,
        expiry_time=t,
        risk_free=RISK_FREE,
        div_yield=DIV_YIELD,
        volatility=sigma,
    )
    return Quote(strike=strike, option_type=OptionType.CALL, price=price(model))


def _two_expiry_surface(t1: float, sig1: float, t2: float, sig2: float) -> VolSurface:
    # Same strike (100) at two maturities, each priced at its own vol.
    return VolSurface(
        spot=SPOT,
        risk_free=RISK_FREE,
        div_yield=DIV_YIELD,
        slices=[
            ExpirySlice(expiry_time=t1, quotes=[_call_quote(100.0, sig1, t1)]),
            ExpirySlice(expiry_time=t2, quotes=[_call_quote(100.0, sig2, t2)]),
        ],
    )


def test_calendar_violation_is_detected() -> None:
    # Short expiry (T=0.5, sigma=0.40 -> w=0.08) carries MORE total variance than
    # the long expiry (T=1.0, sigma=0.20 -> w=0.04) -> calendar arbitrage.
    surface = _two_expiry_surface(t1=0.5, sig1=0.40, t2=1.0, sig2=0.20)

    report = detect(surface)

    cal = next(v for v in report.violations if v.kind == ViolationType.CALENDAR)
    assert cal.magnitude == approx(0.08 - 0.04, abs=1e-4)


def test_increasing_total_variance_is_arbitrage_free() -> None:
    # Total variance rises with maturity (w=0.02 -> w=0.04) -> no calendar arb.
    surface = _two_expiry_surface(t1=0.5, sig1=0.20, t2=1.0, sig2=0.20)

    report = detect(surface)

    assert report.is_arbitrage_free


def test_unpaired_strike_is_skipped() -> None:
    # Only a call at this strike => parity cannot be checked, no violation.
    call = _bs_price(OptionType.CALL, 100.0)
    surface = _surface(
        [Quote(strike=100.0, option_type=OptionType.CALL, price=call)]
    )

    report = detect(surface)

    assert report.is_arbitrage_free
