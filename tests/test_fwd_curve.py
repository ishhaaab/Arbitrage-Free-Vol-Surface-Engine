"""Tests for the forward curve estimator."""
from datetime import date
from math import exp
from pytest import approx

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.repair.fwd_curve import estimate_forward_curve, _slice_forward


SPOT = 100.0
R = 0.05
T = 1.0
_DUMMY_DATE = date(2030, 1, 1)


def _call_price(strike: float, sigma: float = 0.2, tt: float = T) -> float:
    from arbfree_vol.models.option import OptionContract, BlackScholesInput
    from arbfree_vol.pricing.black_scholes import price

    contract = OptionContract(
        symbol="X", option_type=OptionType.CALL, strike=strike,
        expiry_date=_DUMMY_DATE,
    )
    model = BlackScholesInput(
        contract=contract, spot=SPOT, expiry_time=tt,
        risk_free=R, div_yield=0.0, volatility=sigma,
    )
    return price(model)


def _put_price(strike: float, sigma: float = 0.2, tt: float = T) -> float:
    from arbfree_vol.models.option import OptionContract, BlackScholesInput
    from arbfree_vol.pricing.black_scholes import price

    contract = OptionContract(
        symbol="X", option_type=OptionType.PUT, strike=strike,
        expiry_date=_DUMMY_DATE,
    )
    model = BlackScholesInput(
        contract=contract, spot=SPOT, expiry_time=tt,
        risk_free=R, div_yield=0.0, volatility=sigma,
    )
    return price(model)


def test_fwd_curve_recovers_theoretical_forward() -> None:
    # With q=0, the forward should be S * exp(r * T) = 100 * exp(0.05).
    # If both C and P are priced at the same vol (consistent), parity
    # should recover this forward.
    slice_ = ExpirySlice(
        expiry_time=T,
        quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=_call_price(100.0)),
            Quote(strike=100.0, option_type=OptionType.PUT, price=_put_price(100.0)),
            Quote(strike=110.0, option_type=OptionType.CALL, price=_call_price(110.0)),
            Quote(strike=110.0, option_type=OptionType.PUT, price=_put_price(110.0)),
        ],
    )
    surface = VolSurface(spot=SPOT, risk_free=R, div_yield=0.0, slices=[slice_])

    curve = estimate_forward_curve(surface)

    assert T in curve
    assert curve[T] == approx(SPOT * exp(R * T), abs=0.02)  # within 2%


def test_fwd_curve_fallback_when_no_call_put_pairs() -> None:
    # Only calls — no way to extract F from parity, so we fallback to q=0.
    slice_ = ExpirySlice(
        expiry_time=T,
        quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=10.0),
            Quote(strike=110.0, option_type=OptionType.CALL, price=5.0),
        ],
    )
    surface = VolSurface(spot=SPOT, risk_free=R, div_yield=0.0, slices=[slice_])

    curve = estimate_forward_curve(surface)

    assert curve[T] == SPOT * exp(R * T)


def test_fwd_curve_multiple_slices() -> None:
    slices = [
        ExpirySlice(
            expiry_time=0.5,
            quotes=[
                Quote(strike=100.0, option_type=OptionType.CALL, price=_call_price(100.0, sigma=0.2, tt=0.5)),
                Quote(strike=100.0, option_type=OptionType.PUT, price=_put_price(100.0, sigma=0.2, tt=0.5)),
            ],
        ),
        ExpirySlice(
            expiry_time=1.0,
            quotes=[
                Quote(strike=100.0, option_type=OptionType.CALL, price=_call_price(100.0, sigma=0.2)),
                Quote(strike=100.0, option_type=OptionType.PUT, price=_put_price(100.0, sigma=0.2)),
            ],
        ),
    ]
    surface = VolSurface(spot=SPOT, risk_free=R, div_yield=0.0, slices=slices)

    curve = estimate_forward_curve(surface)

    assert sorted(curve.keys()) == [0.5, 1.0]
    for T, F in curve.items():
        assert F == approx(SPOT * exp(R * T), abs=0.02)
