from arbfree_vol.arbitrage.report import ArbitrageReport, ArbitrageViolation, ViolationType
from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType, OffendingQuote
from arbfree_vol.variance import slice_total_variance

from math import exp

# helper funcs:

def _forward(
        surface: VolSurface,
        s: ExpirySlice,
        strike: float) -> float: #returns the fwd price: F = Se^{−qT} − Ke^{−rT}

    return surface.spot * exp(-surface.div_yield * s.expiry_time) \
        - strike * exp(-surface.risk_free * s.expiry_time)



def _check_parity(
        surface:VolSurface,
        s:ExpirySlice,
        violations: list[ArbitrageViolation])-> None: # checks whether there's put call parity over different Ks

    by_strike= {}                       # we want to group options by K, ie K is the key and
    for q in s.quotes:                  # {option_type:price} which is another dict is assigned to K whenever
        by_strike.setdefault(q.strike, {})[q.option_type]= q.price     # theres multiple options with the same K

    for strike, sides in by_strike.items():
        if OptionType.CALL not in sides or OptionType.PUT not in sides:
            continue # we dont use return as it exits the entire function, continue skips an iteration
        C = sides[OptionType.CALL]
        P = sides[OptionType.PUT]

        K= strike
        F= _forward(surface, s, K)


        threshold= 1e-4
        if abs((C-P)-F) > threshold:
            # Both the call and the put at this strike are potentially bad.
            violations.append(ArbitrageViolation(
                kind= ViolationType.PARITY,
                detail=f"put-call parity off at K={K}, T={s.expiry_time}: C-P={C-P:.4f} vs forward={F:.4f}",
                magnitude= float(abs((C-P)-F)),
                offending=(
                    OffendingQuote(strike=K, expiry_time=s.expiry_time, option_type=OptionType.CALL),
                    OffendingQuote(strike=K, expiry_time=s.expiry_time, option_type=OptionType.PUT),
                )))


def _normalize_to_calls(
        surface:VolSurface,
        s:ExpirySlice)-> list[tuple[float, float]]:

    by_strike= {} # again we sort by strike price, K
    for q in s.quotes:
        by_strike.setdefault(q.strike, {})[q.option_type]= q.price


    calls: list[tuple[float, float]] = [] #create an empty list of tuples
    for strike, sides in by_strike.items(): # iterate over K
        if OptionType.CALL in sides:    # if a call exists for some K, we'll j use its price
            call_price = sides[OptionType.CALL]

        else:   # if theres no calls then we convert Put's price into a call Price
            call_price = sides[OptionType.PUT] + _forward(surface, s, strike)

        calls.append((strike, call_price)) # we do above loop for all Ks and turn into a K, Call pricec tuple

    return sorted(calls)


def _check_monotonicity(
    surface: VolSurface,
    s: ExpirySlice,
    calls: list[tuple[float, float]],
    violations: list[ArbitrageViolation]) -> None:
    # Call prices must be non increasing in strike. A strict rise leads to arbitrage.

    for i in range(len(calls) - 1):
        k1, c1= calls[i]
        k2, c2= calls[i + 1]
        jump= c2 - c1

        threshold= 1e-4
        if jump > threshold:
            # The offending call is the one at the higher strike.
            violations.append(ArbitrageViolation(
                kind=ViolationType.MONOTONICITY,
                detail=f"call price rose from {c1:.4f} to {c2:.4f} between K={k1} and K={k2}",
                magnitude=float(jump),
                offending=(
                    OffendingQuote(strike=k2, expiry_time=s.expiry_time, option_type=OptionType.CALL),
                ),
            ))

def _check_butterfly(
    s: ExpirySlice,
    calls: list[tuple[float, float]],
    violations: list[ArbitrageViolation]) -> None:
    # Call prices must convex in strike. We check that via line joining two points test

    for i in range(len(calls)-2):
        k1, c1= calls[i]
        k2, c2= calls[i+1]
        k3, c3= calls[i+2]

        w= (k3-k2)/(k3-k1)
        line= w*c1 + (1-w)*c3

        threshold= 1e-4
        if c2 - line > threshold:
            # The offending call is the one at the middle strike.
            violations.append(ArbitrageViolation(
                kind= ViolationType.BUTTERFLY,
                detail=f"call convexity broken at K={k2}: C={c2:.4f} exceeds line {line:.4f} (from K={k1},{k3})",
                magnitude= float(c2- line),
                offending=(
                    OffendingQuote(strike=k2, expiry_time=s.expiry_time, option_type=OptionType.CALL),
                ),
            ))


def _check_calendar(
    surface: VolSurface,
    violations: list[ArbitrageViolation],
)-> None:

    ordered = sorted(surface.slices, key=lambda sl: sl.expiry_time) # we sort the slices by expiry time

    for i in range(len(ordered)-1):

        earlier= ordered[i]
        later= ordered[i+1]

        total_variance_earlier= slice_total_variance(surface, earlier)
        total_variance_later= slice_total_variance(surface, later)

        for K in total_variance_earlier.keys() & total_variance_later.keys(): # for all common Ks in i, i+1 check for any discrepancy in total variance as a non decreasing func of expiry time
                                                                              # later ill interpolate K instead of finding out common ones.

            diff = total_variance_earlier[K] - total_variance_later[K]

            threshold= 1e-4
            if diff > threshold:
                # Both slices' quotes at this K are offenders.
                violations.append(ArbitrageViolation(
                    kind=ViolationType.CALENDAR,
                    detail=f"calendar arb at K={K}: w={total_variance_earlier[K]:.4f} at T={earlier.expiry_time} exceeds w={total_variance_later[K]:.4f} at T={later.expiry_time}",
                    magnitude=  float(total_variance_earlier[K] - total_variance_later[K]),
                    offending=(
                        OffendingQuote(strike=K, expiry_time=earlier.expiry_time, option_type=OptionType.CALL),
                        OffendingQuote(strike=K, expiry_time=later.expiry_time, option_type=OptionType.CALL),
                    ),
                ))


def _check_wide_spread(
    s: ExpirySlice,
    violations: list[ArbitrageViolation],
    threshold: float= 0.5,
)-> None:
    """Flag quotes whose relative bid-ask spread exceeds threshold.

    Spread is (ask - bid) / mid.  Quotes with no bid/ask data are skipped.
    """
    for q in s.quotes:
        if q.bid is None or q.ask is None:
            continue
        if q.bid <= 0 or q.ask <= 0 or q.bid > q.ask:
            continue
        mid = (q.bid + q.ask) / 2.0
        spread = (q.ask - q.bid) / mid
        if spread > threshold:
            violations.append(ArbitrageViolation(
                kind= ViolationType.WIDE_SPREAD,
                detail= f"wide bid-ask spread at K={q.strike}, T={s.expiry_time}: "
                        f"bid={q.bid:.4f}, ask={q.ask:.4f}, relative spread={spread:.4f}",
                magnitude= float(spread),
                offending=(
                    OffendingQuote(strike=q.strike, expiry_time=s.expiry_time, option_type=q.option_type),
                ),
            ))


def detect(surface:VolSurface) -> ArbitrageReport:

    violations: list[ArbitrageViolation] = []
    for slices in surface.slices:
        _check_parity(surface, slices, violations)
        calls = _normalize_to_calls(surface, slices)
        _check_monotonicity(surface, slices, calls, violations)
        _check_butterfly(slices, calls, violations)
        _check_wide_spread(slices, violations)

    _check_calendar(surface, violations)

    return ArbitrageReport(violations=violations)

