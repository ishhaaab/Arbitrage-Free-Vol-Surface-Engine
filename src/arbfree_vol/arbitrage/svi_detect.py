from arbfree_vol.svi.model import SVIParams
from arbfree_vol.arbitrage.report import ArbitrageViolation, ArbitrageReport, ViolationType
from math import sqrt


def min_total_variance(params: SVIParams)-> float:
    """Returns the minimum total variance ie the min value of the SVI curve
    found at dw(k)/dk=0 where w(k)= a + b(rho(k−m) + √((k−m)**2+sigma**2))
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


def detect_svi(params: SVIParams) -> ArbitrageReport:

    violations: list[ArbitrageViolation] = []
    _check_min_variance(params, violations)
    return ArbitrageReport(violations=violations)