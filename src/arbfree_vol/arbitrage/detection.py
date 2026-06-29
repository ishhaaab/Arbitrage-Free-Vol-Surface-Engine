from arbfree_vol.arbitrage.report import ArbitrageReport, ArbitrageViolation, ViolationType
from arbfree_vol.models.surface import VolSurface, ExpirySlice
from arbfree_vol.models.option import OptionType

from math import exp


def _check_parity(surface:VolSurface,
                   s:ExpirySlice, 
                   violations: list[ArbitrageViolation])-> None:

    by_strike= {}                       # we want to group options by K, ie K is the key and
    for q in s.quotes:                  # {option_type:price} which is another dict is assigned to K whenever 
        by_strike.setdefault(q.strike, {})[q.option_type] = q.price     # theres options with the same K

    for strike, sides in by_strike.items():
        if OptionType.CALL not in sides or OptionType.PUT not in sides:
            continue # we dont use return as it exits the entire function, continue skips this iteration
        C = sides[OptionType.CALL]
        P = sides[OptionType.PUT]

        S= surface.spot
        r= surface.risk_free
        y=surface.div_yield
        T= s.expiry_time
        K= strike
        F= S* exp(-y*T) - K *exp(-r*T)


        threshold= 1e-4
        if abs((C-P)-F) > threshold:
            violations.append(ArbitrageViolation(
                kind= ViolationType.PARITY,
                detail=f"put-call parity off at K={K}, T={T}: C-P={C-P:.4f} vs forward={F:.4f}",
                magnitude= float(abs((C-P)-F))))


        
def detect(surface:VolSurface) -> ArbitrageReport:

    violations: list[ArbitrageViolation] = []
    for slices in surface.slices:
        _check_parity(surface, slices, violations)
    return ArbitrageReport(violations=violations)