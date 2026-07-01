"""Black-Scholes pricing functions."""

from arbfree_vol.models.option import BlackScholesInput, OptionType
from arbfree_vol.pricing._core import core, norm_cdf

def price_floats(
    S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool
) -> float:
    """Float-level Black-Scholes price"""
    c = core(S=S, K=K, T=T, r=r, q=q, sigma=sigma)

    if is_call:
        return S* c.df_q* norm_cdf(c.d1) - K* c.df_r* norm_cdf(c.d2)
    return K* c.df_r* norm_cdf(-c.d2) - S* c.df_q* norm_cdf(-c.d1)


def price(model: BlackScholesInput) -> float:
    is_call = model.contract.option_type == OptionType.CALL
    return price_floats(
        S=model.spot,
        K=model.contract.strike,
        T=model.expiry_time,
        r=model.risk_free,
        q=model.div_yield,
        sigma=model.volatility,
        is_call=is_call,
    )
