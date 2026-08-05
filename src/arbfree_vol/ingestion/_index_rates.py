"""Shared index dividend-yield helpers for the ingestion layer.

Single source of truth for the index representative-ETF mapping and the
per-expiry put-call-parity dividend yield estimators.  Both
``ingestion.yfinance`` and ``ingestion.openbb`` re-import these names so
their call sites keep working unchanged, and so a fix to the estimation
logic lands in exactly one place.

This module does not hard-require ``yfinance`` at import time: it is
imported lazily inside ``_get_representative_dividend_yield`` so that
importing this module (or the ``openbb`` ingestion module that
re-imports its names) does not require ``yfinance`` at import time —
only that fallback path needs it at call time.
"""

import logging

from arbfree_vol.models.surface import ExpirySlice
from arbfree_vol.models.option import OptionType

_logger = logging.getLogger(__name__)


# Mapping of index symbols to representative ETFs that track the same
# (or very similar) underlying basket.  Used as a FALLBACK when per-expiry
# put-call parity estimation of q fails (e.g., no ATM call/put pair).
# This is the ETF's TRAILING yield, not the index's market-implied forward
# yield — it is an approximation.  Per-expiry put-call parity is preferred.
_INDEX_REPRESENTATIVE: dict[str, str | None] = {
    "^SPX": "SPY",   # SPY tracks S&P 500, same constituents
    "^NDX": "QQQ",   # QQQ tracks Nasdaq-100
    "^DJI": "DIA",   # DIA tracks Dow Jones
    "^RUT": "IWM",   # IWM tracks Russell 2000
    "^VIX": None,    # VIX has no constituents
    # Add more as needed
}


def _estimate_index_dividend_yield(
    slice_: ExpirySlice,
    spot: float,
    r: float,
) -> float | None:
    """Estimate the dividend yield for one expiry slice via put-call parity.

    Uses 3-5 strikes on each side of ATM (6-10 strikes total) to solve
    the put-call parity relation for q:

        C - P + K * e^{-rT} = S * e^{-qT}
        q = -log((C - P + K * e^{-rT}) / S) / T

    The wider band (vs. just the 3 nearest strikes) averages out
    single-strike quote noise that dominates the <0.10y bucket where
    the bid-ask spread is widest.

    Returns the MEDIAN q across all usable ATM pairs, or None if
    estimation fails (no call/put pair, or invalid values).
    """
    from statistics import median
    from math import exp, log

    if slice_.expiry_time <= 0:
        return None

    by_strike: dict[float, dict[OptionType, float]] = {}
    for q in slice_.quotes:
        by_strike.setdefault(q.strike, {})[q.option_type] = q.price

    # 3-5 strikes on each side of ATM (6-10 total) for noise robustness.
    # Strike grid on SPX/SPY is typically $1 or $5 wide, so 3-5 strikes on
    # each side spans ~$6-$50 around ATM depending on spacing. This wider
    # band averages out single-strike quote noise that dominates the
    # <0.10y bucket where the bid-ask spread is widest.
    all_strikes_sorted = sorted(by_strike.keys())
    atm_idx = min(range(len(all_strikes_sorted)), key=lambda i: abs(all_strikes_sorted[i] - spot))
    window = 5  # strikes on each side
    low_idx = max(0, atm_idx - window)
    high_idx = min(len(all_strikes_sorted), atm_idx + window + 1)
    atm_strikes = all_strikes_sorted[low_idx:high_idx]

    qs: list[float] = []
    for K in atm_strikes:
        sides = by_strike[K]
        if OptionType.CALL not in sides or OptionType.PUT not in sides:
            continue
        C = sides[OptionType.CALL]
        P = sides[OptionType.PUT]
        T = slice_.expiry_time
        numerator = C - P + K * exp(-r * T)
        if numerator <= 0 or spot <= 0:
            continue
        q_est = -log(numerator / spot) / T
        # Sanity check: dividend yield should be in a reasonable range
        if -0.5 < q_est < 0.5:
            qs.append(q_est)

    if not qs:
        return None
    return float(median(qs))


def _get_representative_dividend_yield(symbol: str) -> float | None:
    """Fetch the trailing dividend yield from a representative ETF for an
    index symbol.

    This is a FALLBACK used only when per-expiry put-call parity
    estimation of q fails.  The returned value is the ETF's trailing
    yield (e.g., SPY's ~1.3%), not the index's market-implied forward
    yield — it is an approximation.  Per-expiry put-call parity is
    preferred because it uses the actual options data.

    Returns None if no representative is mapped, or the representative
    ticker's dividend yield cannot be fetched.
    """
    import yfinance as yf

    rep = _INDEX_REPRESENTATIVE.get(symbol)
    if rep is None:
        return None
    try:
        rep_ticker = yf.Ticker(rep)
        info = rep_ticker.info or {}
        q = info.get("dividendYield")
        if q is not None and isinstance(q, (int, float)) and q > 0:
            q = float(q)
            if q > 0.50:
                q /= 100.0
            return q
    except Exception:
        _logger.warning(
            "Failed to fetch representative dividend yield for %s via %s",
            symbol, rep, exc_info=True,
        )
    return None
