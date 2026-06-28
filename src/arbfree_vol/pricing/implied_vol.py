"""Implied volatility solver."""

from typing import cast

from scipy.optimize import brentq

from arbfree_vol.models.option import BlackScholesInput, ImpliedVolInput
from arbfree_vol.pricing.black_scholes import price


def implied_vol(model: ImpliedVolInput, low: float = 1e-6, high: float = 5.0,) -> float | None:
    base = BlackScholesInput(
        contract=model.contract,
        spot=model.spot,
        expiry_time=model.expiry_time,
        risk_free=model.risk_free,
        div_yield=model.div_yield,
        volatility=1.0,  # seed vol, overwritten each evaluation
    )

    def f(sigma: float) -> float:
        return price(base.model_copy(update={"volatility": sigma})) - model.market_price

    f_low = f(low)
    f_high = f(high)

    # Same sign at both ends then target price is outside [price(low), price(high)],
    # so no implied vol exists in the bracket like an arbitrage vioating quote.
    if f_low * f_high > 0:
        return None
    
    return cast(float, brentq(f, low, high))
