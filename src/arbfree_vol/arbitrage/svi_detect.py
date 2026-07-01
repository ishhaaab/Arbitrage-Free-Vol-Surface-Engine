from arbfree_vol.svi.model import SVIParams, svi_g
from arbfree_vol.arbitrage.report import ArbitrageViolation, ArbitrageReport, ViolationType
from math import sqrt
from scipy.optimize import minimize_scalar


def min_total_variance(params: SVIParams)-> float:
    """Returns the minimum total variance ie the min value of the SVI curve
    found at dw(k)/dk=0 where w(k)= a + b(rho(k-m) + √((k-m)**2+sigma**2))
    that gives us w min= a + b* sigma* sqrt(1-rho**2)"""

    return params.a + params.b* params.sigma* sqrt(1-params.rho**2)


def _check_min_variance(params: SVIParams,
                        violations: list[ArbitrageViolation]) -> None:
    
    total_variance_min= min_total_variance(params)

    tolerance= 1e-4 
    if total_variance_min < - tolerance :
        violations.append(ArbitrageViolation(
            kind= ViolationType.NEGATIVE_VARIANCE,
            detail=f"SVI min total variance is negative: w_min={total_variance_min:.4f}",
            magnitude= -total_variance_min
        ))




def _check_butterfly(params: SVIParams,
                     violations: list[ArbitrageViolation],
                     k_min: float = -2.0,
                    k_max: float = 2.0) -> None:
    
    a, b, rho, m, sigma= params.a, params.b, params.rho, params.m, params.sigma
    
    def single_g(k):
        return svi_g(k, a, b, rho, m, sigma)
        
    
    min_g=  minimize_scalar(fun= single_g, method="bounded", bounds= (k_min,k_max))
    value, location= min_g.fun, min_g.x
    
    tolerance= 1e-4
    if value < -tolerance:
        violations.append(ArbitrageViolation(
            kind=ViolationType.BUTTERFLY,
            detail=f"SVI butterfly arbitrage: g={value:.6f} < 0 at k={location:.4f} (negative risk-neutral density)",
            magnitude= -value
        ))





def detect_svi(params: SVIParams) -> ArbitrageReport:

    violations: list[ArbitrageViolation] = []
    _check_min_variance(params, violations)
    _check_butterfly(params, violations)
    return ArbitrageReport(violations=violations)