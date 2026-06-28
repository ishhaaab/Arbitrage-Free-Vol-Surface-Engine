"""Tests for the implied volatility solver."""

from datetime import date

from pytest import approx

from arbfree_vol.models.option import (
    BlackScholesInput,
    ImpliedVolInput,
    OptionContract,
    OptionType,
)
from arbfree_vol.pricing.black_scholes import price
from arbfree_vol.pricing.implied_vol import implied_vol


def _contract(option_type: OptionType) -> OptionContract:
    return OptionContract(
        symbol="NVDA",
        option_type=option_type,
        strike=100,
        expiry=date(2026, 11, 27),
    )


def _iv_input(option_type: OptionType, market_price: float) -> ImpliedVolInput:
    return ImpliedVolInput(
        contract=_contract(option_type),
        spot=100,
        expiry_time=1,
        risk_free=0.05,
        div_yield=0,
        market_price=market_price,
    )


def test_round_trip_recovers_call_volatility() -> None:
    true_sigma = 0.2
    bs = BlackScholesInput(
        contract=_contract(OptionType.CALL),
        spot=100,
        expiry_time=1,
        risk_free=0.05,
        div_yield=0,
        volatility=true_sigma,
    )
    p = price(bs)

    recovered = implied_vol(_iv_input(OptionType.CALL, p))

    assert recovered == approx(true_sigma, abs=1e-6)


def test_round_trip_recovers_put_volatility() -> None:
    true_sigma = 0.35
    bs = BlackScholesInput(
        contract=_contract(OptionType.PUT),
        spot=100,
        expiry_time=1,
        risk_free=0.05,
        div_yield=0,
        volatility=true_sigma,
    )
    p = price(bs)

    recovered = implied_vol(_iv_input(OptionType.PUT, p))

    assert recovered == approx(true_sigma, abs=1e-6)


def test_price_above_no_arbitrage_bound_returns_none() -> None:
    # A call can never be worth more than the discounted spot; 200 is impossible.
    assert implied_vol(_iv_input(OptionType.CALL, 200.0)) is None


def test_price_below_intrinsic_returns_none() -> None:
    # Deep ITM call: spot 100, strike 50 ie intrinsic ~50. A price of 1.0 is below
    # any achievable model price, so no implied vol exists.
    model = ImpliedVolInput(
        contract=OptionContract(
            symbol="NVDA",
            option_type=OptionType.CALL,
            strike=50,
            expiry=date(2026, 11, 27),
        ),
        spot=100,
        expiry_time=1,
        risk_free=0.05,
        div_yield=0,
        market_price=1.0,
    )
    assert implied_vol(model) is None
