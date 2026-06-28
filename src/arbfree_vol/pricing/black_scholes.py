"""Black-Scholes pricing functions."""

from math import log, sqrt, exp
from scipy.stats import norm

from arbfree_vol.models.option import BlackScholesInput, OptionType

def price(model: BlackScholesInput):

    s= model.spot
    k= model.contract.strike
    r= model.risk_free
    q= model.div_yield
    t= model.expiry_time
    sigma= model.volatility

    d1= (log(s/k) + (r - q + 0.5*sigma**2)*t)/(sigma*sqrt(t))
    d2= d1- sigma*sqrt(t)

    if model.contract.option_type== OptionType.CALL:
        return s * exp(-q*t) * float(norm.cdf(d1)) - k * exp(-r*t) * float(norm.cdf(d2))
    
    elif model.contract.option_type== OptionType.PUT:
        return  k * exp(-r*t) * float(norm.cdf(-d2)) - s * exp(-q*t) * float(norm.cdf(-d1)) 
    
    raise ValueError("Invalid Option Type")
