"""Tests for static arbitrage detection."""

from datetime import date

from pytest import approx

from arbfree_vol.arbitrage.detection import detect
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
        expiry=date(2026, 11, 27),
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


def test_unpaired_strike_is_skipped() -> None:
    # Only a call at this strike => parity cannot be checked, no violation.
    call = _bs_price(OptionType.CALL, 100.0)
    surface = _surface(
        [Quote(strike=100.0, option_type=OptionType.CALL, price=call)]
    )

    report = detect(surface)

    assert report.is_arbitrage_free
