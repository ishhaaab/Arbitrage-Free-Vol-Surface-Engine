from math import sqrt
from pydantic import BaseModel, Field




def svi_total_variance(k: float, a: float, b: float, rho: float, m: float, sigma: float) -> float:

    km= k-m
    total_variance= a+b*((rho*km+ sqrt(km**2 + sigma**2)))
    return total_variance


class SVIParams(BaseModel):
    a: float
    b: float= Field(...,ge=0)
    rho:float= Field(..., gt=-1, lt=1)
    m:float
    sigma:float= Field(...,gt=0)







