"""Black-Scholes Greek calculations."""

from dataclasses import dataclass

from arbfree_vol.models.option import BlackScholesInput, OptionType
from arbfree_vol.pricing._core import core, norm_cdf, norm_pdf

@dataclass(frozen=True, slots=True)
class Greeks:

    delta:float
    gamma: float 
    vega: float 
    theta:float 
    rho: float

def greeks(model: BlackScholesInput) -> Greeks:

    S= model.spot
    K= model.contract.strike
    T= model.expiry_time
    r= model.risk_free
    q= model.div_yield
    sigma= model.volatility
    
    c= core(S=S, K=K, T=T, r=r, q=q, sigma= sigma)
    pdf_d1 = norm_pdf(c.d1)

    sign = 1.0 if model.contract.option_type == OptionType.CALL else -1.0

    delta = c.df_q* sign* norm_cdf(sign* c.d1)
    gamma = c.df_q* pdf_d1 / (S* sigma* c.sqrt_T)
    vega  = S* c.df_q* pdf_d1* c.sqrt_T

    common= -(S* c.df_q* pdf_d1* sigma) / (2* c.sqrt_T)
    term2 = sign* r* K* c.df_r* norm_cdf(sign*c.d2)
    term3= sign* q* S* c.df_q* norm_cdf(sign* c.d1)
    theta = common - term2 + term3

    rho   = sign* K* T* c.df_r* norm_cdf(sign* c.d2)

    return Greeks(delta=delta, gamma= gamma, vega= vega, theta=theta, rho=rho)





