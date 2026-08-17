from math import exp

from arbfree_vol.arbitrage.report import ArbitrageReport, ArbitrageViolation, ViolationType
from arbfree_vol.arbitrage.calendar import _check_calendar
from arbfree_vol.models.surface import VolSurface, ExpirySlice, get_r, get_q
from arbfree_vol.models.option import OptionType, OffendingQuote
from arbfree_vol.forward import estimate_forward_curve, populate_per_slice_r


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


def _group_by_strike(
    s: ExpirySlice,
    *,
    price_only: bool = False,
) -> dict[float, dict[OptionType, object]]:
    """Group a slice's quotes by strike.

    Returns ``{strike: {option_type: value}}`` where ``value`` is the
    Quote (for parity, which needs bid/ask) or the quote's price (for
    synthetic-call normalization).
    """
    by_strike: dict[float, dict[OptionType, object]] = {}
    for q in s.quotes:
        by_strike.setdefault(q.strike, {})[q.option_type] = q.price if price_only else q
    return by_strike


def _check_parity(
        surface: VolSurface,
        s: ExpirySlice,
        violations: list[ArbitrageViolation],  # checks whether there's put call parity over different Ks
        forward_price: float | None = None) -> None:
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
        is the sum of the two half-spreads (capped at a $0.05
        floor).  This treats residuals inside the combined
        execution cost as market noise, not arb.
      - If bid/ask is missing, a fixed $0.05 threshold is used.
        This is calibrated for liquid US equities and ETFs (SPY, QQQ,
        AAPL, NVDA, MSFT) where prices are >= $0.05 and spreads
        are typically < $0.10.  For index options (SPX, NDX) with
        $0.10--$0.30 spreads, bump it to $0.10--$0.15; for illiquid
        names with $0.50+ spreads, use a larger value.
    """
    by_strike = _group_by_strike(s)

    for strike, sides in by_strike.items():
        if OptionType.CALL not in sides or OptionType.PUT not in sides:
            continue
        C_q = sides[OptionType.CALL]
        P_q = sides[OptionType.PUT]
        C = C_q.price
        P = P_q.price

        K = strike
        if forward_price is not None:
            # Use the explicit forward:  C - P = e^{-rT}(F - K)
            r = get_r(surface, s)
            F = forward_price
            parity_rhs = exp(-r * s.expiry_time) * (F - K)
        else:
            # Fall back to surface-level r/q
            parity_rhs = _parity_rhs(surface, s, K)

        threshold = _parity_threshold(C_q, P_q)

        if abs((C - P) - parity_rhs) > threshold:
            # Both the call and the put at this strike are potentially bad.
            violations.append(ArbitrageViolation(
                kind=ViolationType.PARITY,
                detail=f"put-call parity off at K={K}, T={s.expiry_time}: C-P={C-P:.4f} vs RHS={parity_rhs:.4f}",
                magnitude=float(abs((C - P) - parity_rhs)),
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
    by_strike = _group_by_strike(s, price_only=True)

    r = get_r(surface, s)

    calls: list[tuple[float, float]] = []  # creates an empty list of tuples
    for strike, sides in by_strike.items():  # iterate over K
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
        k1, c1 = calls[i]
        k2, c2 = calls[i + 1]
        jump = c2 - c1

        threshold = 1e-4
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

    for i in range(len(calls) - 2):
        k1, c1 = calls[i]
        k2, c2 = calls[i + 1]
        k3, c3 = calls[i + 2]

        w = (k3 - k2) / (k3 - k1)
        line = w * c1 + (1 - w) * c3

        threshold = 1e-4
        if c2 - line > threshold:
            # The offending call is the one at the middle strike.
            violations.append(ArbitrageViolation(
                kind=ViolationType.BUTTERFLY,
                detail=f"call convexity broken at K={k2}: C={c2:.4f} exceeds line {line:.4f} (from K={k1},{k3})",
                magnitude=float(c2 - line),
                offending=(
                    OffendingQuote(strike=k2, expiry_time=s.expiry_time, option_type=OptionType.CALL),
                ),
            ))


def _check_wide_spread(s: ExpirySlice,
                       violations: list[ArbitrageViolation],
                       threshold: float = 0.5) -> None:
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
                kind=ViolationType.WIDE_SPREAD,
                detail=f"wide bid-ask spread at K={q.strike}, T={s.expiry_time}: "
                        f"bid={q.bid:.4f}, ask={q.ask:.4f}, relative spread={spread:.4f}",
                magnitude=float(spread),
                offending=(
                    OffendingQuote(strike=q.strike, expiry_time=s.expiry_time, option_type=q.option_type),
                ),
            ))


def _detect_core(
    surface: VolSurface,
    forward_prices: dict[float, float] | None,
) -> ArbitrageReport:
    """Run the per-slice detection checks, optionally threaded with forwards.

    Shared body of ``detect()`` and ``detect_with_forward()``: each slice
    gets parity (with the estimated forward when provided), the
    monotonicity/butterfly checks on synthetic calls, and the wide-spread
    check.  The calendar check runs once across the whole surface.
    """
    violations: list[ArbitrageViolation] = []
    for sl in surface.slices:
        F = None if forward_prices is None else forward_prices.get(sl.expiry_time)
        _check_parity(surface, sl, violations, forward_price=F)
        calls = _normalize_to_calls(surface, sl, forward_price=F)
        _check_monotonicity(surface, sl, calls, violations)
        _check_butterfly(sl, calls, violations)
        _check_wide_spread(sl, violations)

    _check_calendar(surface, violations)

    return ArbitrageReport(violations=violations)


def detect(surface: VolSurface) -> ArbitrageReport:
    """Detect all no-arbitrage violations on a volatility surface.

    Uses surface-level ``r`` and ``q`` for the parity check.  For
    real market data where these constants may be inaccurate, use
    ``detect_with_forward()`` instead — it estimates per-expiry
    forward prices as a pre-pass and feeds them into the parity check.
    """
    return _detect_core(surface, None)


def detect_with_forward(surface: VolSurface) -> ArbitrageReport:
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

    return _detect_core(surface, fwd_curve)
