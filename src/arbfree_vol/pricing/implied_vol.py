"""Implied volatility solver."""

from typing import cast

from scipy.optimize import brentq

from arbfree_vol.models.option import ImpliedVolInput, OptionType
from arbfree_vol.pricing.black_scholes import price_floats


def implied_vol(model: ImpliedVolInput, low: float = 1e-6, high: float = 5.0,) -> float | None:
    S = model.spot
    K = model.contract.strike
    T = model.expiry_time
    r = model.risk_free
    q = model.div_yield
    is_call = model.contract.option_type == OptionType.CALL
    target = model.market_price

    def f(sigma: float) -> float:
        return price_floats(S, K, T, r, q, sigma, is_call) - target

    f_low = f(low)
    f_high = f(high)

    # Same sign at both ends then target price is outside [price(low), price(high)],
    # so no implied vol exists in the bracket like an arbitrage vioating quote.
    if f_low * f_high > 0:
        return None
    
    return cast(float, brentq(f, low, high))
