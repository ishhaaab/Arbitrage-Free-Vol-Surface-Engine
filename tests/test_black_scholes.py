"""Tests for Black-Scholes pricing."""

from datetime import date

from pytest import approx

from arbfree_vol.models.option import BlackScholesInput, OptionContract, OptionType
from arbfree_vol.pricing.black_scholes import price


def test_call_price_matches_known_black_scholes_value() -> None:
    contract = OptionContract(
        symbol="NVDA",
        option_type=OptionType.CALL,
        strike=100,
        expiry_date=date(2026, 11, 27),
    )
    model = BlackScholesInput(
        contract=contract,
        spot=100,
        expiry_time=1,
        risk_free=0.05,
        div_yield=0,
        volatility=0.2,
    )

    assert price(model) == approx(10.4506, abs=1e-4)


def test_put_price_matches_known_black_scholes_value() -> None:
    contract = OptionContract(
        symbol="NVDA",
        option_type=OptionType.PUT,
        strike=100,
        expiry_date=date(2026, 11, 27),
    )
    model = BlackScholesInput(
        contract=contract,
        spot=100,
        expiry_time=1,
        risk_free=0.05,
        div_yield=0,
        volatility=0.2,
    )

    assert price(model) == approx(5.5735, abs=1e-4)
