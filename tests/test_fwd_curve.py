from datetime import date
from math import exp

from pytest import approx

from arbfree_vol.forward import estimate_forward_curve
from arbfree_vol.models.option import BlackScholesInput, OptionContract, OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface
from arbfree_vol.pricing.black_scholes import price


def _option_price(kind: OptionType, strike: float, maturity: float = 1.0) -> float:
    return price(
        BlackScholesInput(
            contract=OptionContract(
                symbol="X", option_type=kind, strike=strike, expiry_date=date(2030, 1, 1)
            ),
            spot=100,
            expiry_time=maturity,
            risk_free=0.05,
            div_yield=0.01,
            volatility=0.2,
        )
    )


def test_median_parity_forward_is_robust_to_one_outlier() -> None:
    quotes: list[Quote] = []
    for strike in (90.0, 100.0, 110.0):
        call = _option_price(OptionType.CALL, strike)
        if strike == 110:
            call += 20
        quotes.extend(
            [
                Quote(strike=strike, option_type=OptionType.CALL, price=call),
                Quote(strike=strike, option_type=OptionType.PUT, price=_option_price(OptionType.PUT, strike)),
            ]
        )
    surface = VolSurface(
        spot=100,
        risk_free=0.05,
        div_yield=0.01,
        slices=[ExpirySlice(expiry_time=1, quotes=quotes)],
    )
    assert estimate_forward_curve(surface)[1] == approx(100 * exp(0.04), abs=1e-10)


def test_forward_uses_documented_carry_when_no_pair_exists() -> None:
    surface = VolSurface(
        spot=100,
        risk_free=0.05,
        div_yield=0.01,
        slices=[
            ExpirySlice(
                expiry_time=0.5,
                quotes=[Quote(strike=100, option_type=OptionType.CALL, price=8)],
            )
        ],
    )
    assert estimate_forward_curve(surface)[0.5] == approx(100 * exp(0.04 * 0.5))
