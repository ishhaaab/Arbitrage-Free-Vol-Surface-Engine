from dataclasses import dataclass

from math import log, sqrt, exp
from scipy.stats import norm

@dataclass(frozen=True, slots=True)
class BSCore:
    d1: float
    d2: float
    df_r: float
    df_q: float
    pdf_d1: float
    sqrt_T: float

def core(S:float, K: float, T: float, r:float, q:float, sigma:float ) -> BSCore:

    sqrt_T= sqrt(T)
    d1= (log(S/K) + (r - q + 0.5*sigma**2)*T)/(sigma*sqrt_T)
    d2= d1- sigma*sqrt_T
    df_r= exp(-r*T)
    df_q= exp(-q*T)
    pdf_d1= float(norm.pdf(d1))

    return BSCore(d1=d1, d2=d2, df_r=df_r, df_q=df_q, pdf_d1=pdf_d1, sqrt_T=sqrt_T)
