"""Black-Scholes pricing functions."""

from scipy.stats import norm

from arbfree_vol.models.option import BlackScholesInput, OptionType
from arbfree_vol.pricing._core import core

def price(model: BlackScholesInput) -> float:

    s= model.spot
    k= model.contract.strike
    t= model.expiry_time
    r= model.risk_free
    q= model.div_yield
    sigma= model.volatility

    c= core(S=s,K=k,T=t,r=r,q=q,sigma=sigma)

    if model.contract.option_type== OptionType.CALL:
        return s * c.df_q * float(norm.cdf(c.d1)) - k * c.df_r * float(norm.cdf(c.d2))
    
    elif model.contract.option_type== OptionType.PUT:
        return  k * c.df_r * float(norm.cdf(-c.d2)) - s * c.df_q * float(norm.cdf(-c.d1)) 
    
    raise ValueError("Invalid Option Type")
