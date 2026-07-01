from dataclasses import dataclass

from math import erf, exp, log, pi, sqrt

_INV_SQRT2 = 1.0 / sqrt(2.0)
_INV_SQRT_2PI = 1.0 / sqrt(2.0 * pi)


def norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf """
    #this was ~100x faster per scalar than scipy.stats.norm.cdf.
    return 0.5 * (1.0 + erf(x * _INV_SQRT2))


def norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return _INV_SQRT_2PI * exp(-0.5 * x * x)


@dataclass(frozen=True, slots=True)
class BScore:
    d1: float
    d2: float
    df_r: float
    df_q: float
    sqrt_T: float


def core(S: float, K: float, T: float, r: float, q: float, sigma: float) -> BScore:

    sqrt_T = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    df_r = exp(-r * T)
    df_q = exp(-q * T)

    return BScore(d1=d1, d2=d2, df_r=df_r, df_q=df_q, sqrt_T=sqrt_T)
