from arbfree_vol.arbitrage.report import ArbitrageReport, ArbitrageViolation, ViolationType
from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote, get_r, get_q
from arbfree_vol.models.option import OptionType, OffendingQuote
from arbfree_vol.variance import slice_total_variance
from arbfree_vol.repair.fwd_curve import estimate_forward_curve

from math import exp

# helper funcs:

def _forward(
        surface: VolSurface,
        s: ExpirySlice,
        strike: float) -> float: #returns the fwd price: F = Se^{−qT} − Ke^{−rT}

    r = get_r(surface, s)
    q = get_q(surface, s)
    return surface.spot * exp(-q * s.expiry_time) \
        - strike * exp(-r * s.expiry_time)



def _check_parity(
        surface:VolSurface,
        s:ExpirySlice,
        violations: list[ArbitrageViolation], # checks whether there's put call parity over different Ks
        forward_price: float | None= None)-> None:
    """Check put-call parity across strikes in a single expiry slice.

    The parity residual is |C - P - RHS| where RHS depends on whether
    an explicit forward price is available:

      - If forward_price is given (preferred), parity is evaluated
        as:

            C - P = e^{-rT} * (F - K)

        where F is the estimated forward from the market (e.g.
        from put-call parity on a richer slice).  This avoids
        surface-level r/q approximations.

      - If forward_price is None (fallback), parity is
        evaluated with the surface-level r and q:

            C - P = S * e^{-qT} - K * e^{-rT}

    Threshold logic:
      - If both the call and put have bid/ask data, the threshold
        is the wider half spread (capped at a $0.05 floor).  This
        treats residuals inside the spread as market noise, not arb.
      - If bid/ask is missing, a fixed $0.05 threshold is used.
        This is calibrated for liquid US equities and ETFs (SPY, QQQ,
        AAPL, NVDA, MSFT) where prices are >= $0.05 and spreads
        are typically < $0.10.  For index options (SPX, NDX) with
        $0.10--$0.30 spreads, bump it to $0.10--$0.15; for illiquid
        names with $0.50+ spreads, use a larger value.
    """
    by_strike= {}                       # we want to group options by K, ie K is the key and
    for q in s.quotes:                  # {option_type:price} which is another dict is assigned to K whenever
                                        # theres multiple options with the same K
        by_strike.setdefault(q.strike, {})[q.option_type]= q     # this gives access to price, bid, ask

    for strike, sides in by_strike.items():
        if OptionType.CALL not in sides or OptionType.PUT not in sides:
            continue
        C_q = sides[OptionType.CALL]
        P_q = sides[OptionType.PUT]
        C = C_q.price
        P = P_q.price

        K= strike
        if forward_price is not None:
            # Use the explicit forward:  C - P = e^{-rT}(F - K)
            r = get_r(surface, s)
            F = forward_price
            parity_rhs = exp(-r * s.expiry_time) * (F - K)
        else:
            # Fall back to surface-level r/q
            parity_rhs = _forward(surface, s, K)

        # compute a market-aware threshold
        if C_q.bid is not None and C_q.ask is not None and P_q.bid is not None and P_q.ask is not None:
            half_spread_C = 0.5 * (C_q.ask - C_q.bid)
            half_spread_P = 0.5 * (P_q.ask - P_q.bid)
            threshold = max(half_spread_C, half_spread_P, 0.05)
        else:
            # fallback for data without bid/ask — calibrated for
            # liquid US equities / ETFs (SPY, QQQ, AAPL, NVDA, MSFT).
            # Adjust to $0.10-$0.15 for index options (SPX, NDX)
            # or larger for illiquid names.
            threshold = 0.05

        if abs((C-P)-parity_rhs) > threshold:
            # Both the call and the put at this strike are potentially bad.
            violations.append(ArbitrageViolation(
                kind= ViolationType.PARITY,
                detail=f"put-call parity off at K={K}, T={s.expiry_time}: C-P={C-P:.4f} vs RHS={parity_rhs:.4f}",
                magnitude= float(abs((C-P)-parity_rhs)),
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
    """Call prices must be non-increasing in strike.  A strict rise is arbitrage."""

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
    """Call prices must be convex in strike.  Violation means the middle call lies above the line joining its neighbours."""

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
    """Total variance must be non-decreasing with time at every common strike."""
    ordered = sorted(surface.slices, key=lambda sl: sl.expiry_time)

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
    """Detect all no-arbitrage violations on a volatility surface.

    Uses surface-level ``r`` and ``q`` for the parity check.  For
    real market data where these constants may be inaccurate, use
    ``detect_with_forward()`` instead — it estimates per-expiry
    forward prices as a pre-pass and feeds them into the parity check.
    """
    violations: list[ArbitrageViolation] = []
    for slices in surface.slices:
        _check_parity(surface, slices, violations)
        calls = _normalize_to_calls(surface, slices)
        _check_monotonicity(surface, slices, calls, violations)
        _check_butterfly(slices, calls, violations)
        _check_wide_spread(slices, violations)

    _check_calendar(surface, violations)

    return ArbitrageReport(violations=violations)


def detect_with_forward(surface:VolSurface) -> ArbitrageReport:
    """Like detect() but uses an estimated forward curve as a pre-pass.

    Runs ``estimate_forward_curve`` to obtain per-expiry forward
    prices from put-call parity, then threads them into the parity
    check.  This prevents systematic false positives when the
    surface-level risk-free rate or dividend yield are inaccurate.

    Recommended for real market data (yfinance, CBOE, etc.).
    Synthetic / test data can safely use ``detect()``.
    """
    fwd_curve = estimate_forward_curve(surface)

    violations: list[ArbitrageViolation] = []
    for sl in surface.slices:
        F = fwd_curve.get(sl.expiry_time)
        _check_parity(surface, sl, violations, forward_price=F)
        calls = _normalize_to_calls(surface, sl)
        _check_monotonicity(surface, sl, calls, violations)
        _check_butterfly(sl, calls, violations)
        _check_wide_spread(sl, violations)

    _check_calendar(surface, violations)

    return ArbitrageReport(violations=violations)

