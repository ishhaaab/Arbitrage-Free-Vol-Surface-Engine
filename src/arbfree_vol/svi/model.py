from math import sqrt
from pydantic import BaseModel, Field
from dataclasses import dataclass

class SVIParams(BaseModel):
    a: float
    b: float= Field(...,ge=0)
    rho:float= Field(..., gt=-1, lt=1)
    m:float
    sigma:float= Field(...,gt=0)

@dataclass(frozen=True, slots=True)
class SVIcore:
    w0: float
    w1: float
    w2: float

def svi_core(k, a, b, rho, m, sigma) -> SVIcore:

    u= k-m
    R= sqrt(u**2+ sigma**2)
    w0= a + b* (rho*u + R)
    w1= b* (rho + u/R)
    w2= b* sigma**2 / R**3

    return SVIcore(w0=w0, w1=w1, w2=w2) 


def svi_g(k, a, b, rho, m, sigma) -> float:

    derivative= svi_core(k, a, b, rho, m, sigma)
    if derivative.w0 <= 0.0:
        return float("-inf")
    g= (1 - k* derivative.w1/(2* derivative.w0))**2 - (derivative.w1**2 /4)*(1/derivative.w0 + 1/4) + derivative.w2/2

    return g



def svi_total_variance(k: float, a: float, b: float, rho: float, m: float, sigma: float) -> float:

    km= k-m
    total_variance= a+b*((rho*km+ sqrt(km**2 + sigma**2)))
    return total_variance

