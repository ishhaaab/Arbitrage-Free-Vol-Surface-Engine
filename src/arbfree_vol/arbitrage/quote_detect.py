from arbfree_vol.arbitrage.report import ArbitrageReport, ArbitrageViolation, ViolationType
from arbfree_vol.models.surface import VolSurface, ExpirySlice, get_r, get_q
from arbfree_vol.models.option import OptionType, OffendingQuote
from arbfree_vol.variance import slice_total_variance
from arbfree_vol.forward import estimate_forward_curve, populate_per_slice_r

from math import exp, log
import numpy as np

# helper funcs:

def _parity_rhs(
        surface: VolSurface,
        s: ExpirySlice,
        strike: float) -> float:
    """Present value of (F - K) under put-call parity.

    Returns e^{-rT}(F - K) = S * e^{-qT} - K * e^{-rT},
    which is the right-hand side of C - P = e^{-rT}(F - K).
    """

    r = get_r(surface, s)
    q = get_q(surface, s)
    return surface.spot * exp(-q * s.expiry_time) \
        - strike * exp(-r * s.expiry_time)



def _parity_threshold(C_q, P_q) -> float:
    """Market-aware parity threshold from the two sides' bid/ask spreads.

    Both spreads must be crossed to execute a parity arbitrage (buy one
    side at ask, sell the other at bid), so the combined execution cost is
    the sum of the half-spreads, not the wider of the two.
    """
    if C_q.bid is not None and C_q.ask is not None and P_q.bid is not None and P_q.ask is not None:
        half_spread_C = 0.5 * (C_q.ask - C_q.bid)
        half_spread_P = 0.5 * (P_q.ask - P_q.bid)
        return max(half_spread_C + half_spread_P, 0.05)

    # fallback for data without bid/ask — calibrated for
    # liquid US equities / ETFs (SPY, QQQ, AAPL, NVDA, MSFT).
    # Adjust to $0.10-$0.15 for index options (SPX, NDX)
    # or larger for illiquid names.
    return 0.05



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
            parity_rhs = _parity_rhs(surface, s, K)

        threshold = _parity_threshold(C_q, P_q)

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
        surface: VolSurface,
        s: ExpirySlice,
        forward_price: float | None = None) -> list[tuple[float, float]]:
    """Convert all quotes in a slice to synthetic call prices.

    When no call exists at a strike, a put is converted via put-call
    parity.  When both exist, the call price is averaged with the
    parity-implied call from the put.

    If *forward_price* is provided (preferred for real market data),
    parity-implied calls use ``P + e^{-rT}(F - K)`` — the estimated
    market forward from put-call parity.  Otherwise falls back to the
    surface-level ``r``/``q`` via ``_parity_rhs``.
    """

    by_strike: dict[float, dict[OptionType, float]] = {}  # again we sort by strike price, K
    for q in s.quotes:
        by_strike.setdefault(q.strike, {})[q.option_type] = q.price

    r = get_r(surface, s)

    calls: list[tuple[float, float]] = [] # creates an empty list of tuples
    for strike, sides in by_strike.items(): # iterate over K
        call_price = _synthetic_call_price(surface, s, strike, sides, forward_price, r)
        calls.append((strike, call_price))

    return sorted(calls)


def _synthetic_call_price(
    surface: VolSurface,
    s: ExpirySlice,
    strike: float,
    sides: dict[OptionType, float],
    forward_price: float | None,
    r: float,
) -> float:
    """Synthetic call price at one strike, from calls and/or put-call parity.

    When a call exists, its price is used.  If a put also exists, the
    call is averaged with the parity-implied call from the put.  When no
    call exists, the put is converted via put-call parity.
    """
    if OptionType.CALL in sides:  # if a call exists for some K, use it
        call_price = sides[OptionType.CALL]
        if OptionType.PUT in sides:  # also have a put then average with parity-implied call
            parity_call = _parity_implied_call(surface, s, strike, sides[OptionType.PUT], forward_price, r)
            call_price = (call_price + parity_call) / 2.0
        return call_price

    # No call at this strike — convert the put via put-call parity.
    return _parity_implied_call(surface, s, strike, sides[OptionType.PUT], forward_price, r)


def _parity_implied_call(
    surface: VolSurface,
    s: ExpirySlice,
    strike: float,
    put_price: float,
    forward_price: float | None,
    r: float,
) -> float:
    """Implied call price from put-call parity: P + e^{-rT}(F - K).

    Uses the explicit forward when given, otherwise falls back to the
    surface-level ``r``/``q`` via ``_parity_rhs``.
    """
    if forward_price is not None:
        return put_price + exp(-r * s.expiry_time) * (forward_price - strike)
    return put_price + _parity_rhs(surface, s, strike)


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


def _check_calendar(surface: VolSurface,
                    violations: list[ArbitrageViolation],
                    n_k: int = 61) -> None:
    """Total variance must be non-decreasing with time at every log-moneyness k.

    Converts each slice to (k, w) space using its per-slice forward price,
    then interpolates onto a common k-grid per adjacent pair.  Flags
    contiguous bands where w_earlier(k) > w_later(k) beyond tolerance.
    """
    ordered = sorted(surface.slices, key=lambda sl: sl.expiry_time)
    tolerance = 1e-4

    for i in range(len(ordered) - 1):
        earlier = ordered[i]
        later = ordered[i + 1]

        grids = _calendar_pair_grids(surface, earlier, later, n_k)
        if grids is None:
            continue

        k_grid, gap = grids
        _append_calendar_violations(
            violations, earlier, later, k_grid, gap, tolerance)


def _calendar_pair_grids(
    surface: VolSurface,
    earlier: ExpirySlice,
    later: ExpirySlice,
    n_k: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Interpolate both slices' total variance onto a common k-grid.

    Returns ``(k_grid, gap)`` where ``gap = w_earlier - w_later`` evaluated
    on the shared log-moneyness grid, or ``None`` when the pair has no
    usable overlap (fewer than 2 points on either side, or disjoint k
    ranges).
    """
    from arbfree_vol.svi.data import _forward_price

    F_e = _forward_price(surface, earlier)
    F_l = _forward_price(surface, later)

    w_e = slice_total_variance(surface, earlier)
    w_l = slice_total_variance(surface, later)

    ew = sorted([(log(K / F_e), w) for K, w in w_e.items()])
    lw = sorted([(log(K / F_l), w) for K, w in w_l.items()])

    if len(ew) < 2 or len(lw) < 2:
        return None

    ks_e, vs_e = zip(*ew)
    ks_l, vs_l = zip(*lw)

    k_min = max(min(ks_e), min(ks_l))
    k_max = min(max(ks_e), max(ks_l))
    if k_min >= k_max:
        return None

    k_grid = np.linspace(k_min, k_max, n_k)
    w_e_interp = np.interp(k_grid, ks_e, vs_e)
    w_l_interp = np.interp(k_grid, ks_l, vs_l)
    return k_grid, w_e_interp - w_l_interp


def _append_calendar_violations(
    violations: list[ArbitrageViolation],
    earlier: ExpirySlice,
    later: ExpirySlice,
    k_grid: np.ndarray,
    gap: np.ndarray,
    tolerance: float,
) -> None:
    """Flag contiguous runs where ``gap > tolerance`` as calendar arbitrage.

    A run is a maximal set of adjacent grid points with ``gap > tolerance``;
    each run becomes one CALENDAR violation naming the pair's expiries, the
    run's k-range, and its worst gap.
    """
    in_run = False
    run_start = 0
    max_gap = 0.0

    for j in range(len(k_grid)):
        if gap[j] > tolerance:
            if not in_run:
                run_start = j
                max_gap = gap[j]
                in_run = True
            else:
                max_gap = max(max_gap, gap[j])
        else:
            if in_run:
                violations.append(_calendar_violation(
                    earlier, later, k_grid, run_start, j - 1, max_gap))
                in_run = False

    if in_run:
        violations.append(_calendar_violation(
            earlier, later, k_grid, run_start, len(k_grid) - 1, max_gap))


def _calendar_violation(
    earlier: ExpirySlice,
    later: ExpirySlice,
    k_grid: np.ndarray,
    i_start: int,
    i_end: int,
    max_gap: float,
) -> ArbitrageViolation:
    """Build one CALENDAR violation for a contiguous run ``[i_start, i_end]``."""
    return ArbitrageViolation(
        kind=ViolationType.CALENDAR,
        detail=f"calendar arb: T={earlier.expiry_time:.4f} > T={later.expiry_time:.4f}, "
                f"k=[{k_grid[i_start]:.4f}, {k_grid[i_end]:.4f}], "
                f"worst gap={max_gap:.6f}",
        magnitude=max_gap,
        offending=(),
    )


def _check_wide_spread(s: ExpirySlice,
                       violations: list[ArbitrageViolation],
                       threshold: float= 0.5)-> None:
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

    Does NOT mutate the input surface: it operates on a deep copy,
    and the forward-implied per-slice rate is a detection-time
    correction only.
    """
    # Work on a deep copy so the caller's surface is never mutated:
    # populate_per_slice_r writes the forward-implied per-slice risk-free
    # rate into each slice, which is a detection-time correction only.
    surface = surface.model_copy(deep=True)
    fwd_curve = estimate_forward_curve(surface)
    populate_per_slice_r(surface, fwd_curve)

    violations: list[ArbitrageViolation] = []
    for sl in surface.slices:
        F = fwd_curve.get(sl.expiry_time)
        _check_parity(surface, sl, violations, forward_price=F)
        calls = _normalize_to_calls(surface, sl, forward_price=F)
        _check_monotonicity(surface, sl, calls, violations)
        _check_butterfly(sl, calls, violations)
        _check_wide_spread(sl, violations)

    _check_calendar(surface, violations)

    return ArbitrageReport(violations=violations)

