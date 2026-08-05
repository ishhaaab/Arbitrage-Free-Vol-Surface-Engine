"""Fetch live option chains from yfinance and build a VolSurface.

Attempts to source real risk-free rates and dividend yields.  Falls
back to pre-pass forward-curve estimation when rates are unavailable.

Index symbols (tickers starting with ``^``, e.g. ``^SPX``) use a
put-call-parity implied dividend yield per expiry (via
``estimate_forward_curve`` / parity q), with a representative-ETF
fallback (``_INDEX_REPRESENTATIVE``, e.g. ``^SPX`` -> SPY) when parity
estimation fails.  Per-slice q choices are logged.
"""

import logging
import math
import warnings
from datetime import date
from typing import Any

import yfinance as yf

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.repair.fwd_curve import estimate_forward_curve
from arbfree_vol.ingestion.cleaning import clean_quotes, RejectionRecord
from arbfree_vol.ingestion._index_rates import (
    _INDEX_REPRESENTATIVE,
    _estimate_index_dividend_yield,
    _get_representative_dividend_yield,
)
from arbfree_vol.data.quality import (
    DataQualityConfig,
    DropRecord,
    filter_option_chain,
)
from arbfree_vol.data.snapshot_guard import check_snapshot_time

_logger = logging.getLogger(__name__)


def _get_risk_free_rate() -> float | None:
    """Fetch the 13-week Treasury yield (^IRX) as a decimal.

    Returns None if the ticker is unavailable or the value is zero / None.
    """
    try:
        irx = yf.Ticker("^IRX")
        info = irx.info or {}
        rate = info.get("regularMarketPrice") or info.get("previousClose")
        if rate is not None and isinstance(rate, (int, float)) and rate > 0:
            return rate / 100.0  # convert percent to decimal
    except Exception:
        _logger.warning("Failed to fetch risk-free rate from ^IRX", exc_info=True)
    return None


def _get_dividend_yield(ticker: yf.Ticker) -> float | None:
    """Fetch the dividend yield from a yfinance ticker info.

    yfinance returns it as a fraction (e.g. 0.013 for 1.3%).
    Returns None if unavailable.
    """
    try:
        info = ticker.info or {}
        q = info.get("dividendYield")
        if q is not None and isinstance(q, (int, float)) and q > 0:
            q = float(q)
            # yfinance sometimes returns percent (1.01 for 1.01%) and
            # sometimes fraction (0.0101).  A yield above 50% is
            # definitely in percent then divide by 100.
            if q > 0.50:
                q /= 100.0
            return q
    except Exception:
        _logger.warning("Failed to fetch dividend yield", exc_info=True)
    return None


def _row_to_quote(row: Any, otype: OptionType) -> Quote | None:
    """Convert a yfinance DataFrame row to a Quote.

    Uses the **mid price** (``(bid + ask) / 2``) when both bid and ask
    are available — this reflects the live market, not stale ``lastPrice``.
    Falls back to ``lastPrice`` if either bid or ask is missing.
    Returns ``None`` when no valid price can be determined.
    """
    def _val(key: str) -> float | None:
        v = row.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)

    bid = _val("bid")
    ask = _val("ask")
    last = _val("lastPrice")

    if bid is not None and ask is not None:
        price = (bid + ask) / 2.0
    elif last is not None:
        price = last
    else:
        return None

    return Quote(
        strike=float(row["strike"]),
        option_type=otype,
        price=price,
        bid=bid,
        ask=ask,
    )


def fetch_chain(
    symbol: str,
    max_expiries: int = 5,
    min_T_years: float = 7.0 / 365.0,
    quality_config: DataQualityConfig | None = None,
    disable_quality_filter: bool = False,
) -> tuple[VolSurface, list[RejectionRecord], list[DropRecord]]:
    """Fetch an option chain from yfinance and return a cleaned VolSurface.

    Steps through the nearest expiries, builds quotes from mid prices
    (not stale lastPrice), applies the cleaning layer (wide spreads,
    crossed markets, deep moneyness, near-expiry, zero prices), and
    returns the cleaned surface plus an audit trail of rejects.

    Real r comes from ``^IRX`` (13-week T-bill), real q from
    ``info.dividendYield``.  When either is unavailable, defaults to
    ``r=0.05, q=0.0`` — the repair pipeline's ``detect_with_forward()``
    corrects for that at detection time.

    When ``quality_config`` is provided (or defaults are used), a
    pre-ingestion data-quality filter is applied to each expiry's raw
    option chain DataFrame *before* building ``Quote`` objects.  Strikes
    failing any threshold (min open interest, max bid-ask spread) are
    dropped and recorded in the returned ``quality_drops`` list.  Volume
    is recorded for diagnostic context but is not a filter criterion.

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"SPY"``).
    max_expiries:
        Maximum number of expiries to process.
    min_T_years:
        Minimum time-to-expiry in years.
    quality_config:
        Data-quality filter thresholds.  Uses ``DataQualityConfig()``
        defaults when ``None`` and ``disable_quality_filter`` is False.
    disable_quality_filter:
        When ``True``, skip the data-quality filter entirely and return
        raw yfinance data.  This is the ONLY way to get truly unfiltered
        data — passing ``quality_config=None`` with
        ``disable_quality_filter=False`` still applies default thresholds.
    """
    # Snapshot-time guard (warns, does not block)
    guard_warning = check_snapshot_time()
    if guard_warning:
        warnings.warn(guard_warning, stacklevel=2)

    ticker = yf.Ticker(symbol)
    expiries = ticker.options

    if not expiries:
        raise ValueError(f"No expiries available for symbol {symbol!r}")

    # source rates
    r = _get_risk_free_rate()
    if r is None:
        _logger.warning(
            "Risk-free rate unavailable for %s (^IRX fetch failed or "
            "empty); substituting r=0.05",
            symbol,
        )
        r = 0.05
    _is_index = symbol.startswith("^")
    if _is_index:
        # For index symbols (^SPX, etc.), estimate q per-expiry via put-call
        # parity.  This is more accurate than hardcoding q=0 because indices
        # have a genuine implied dividend yield from their constituents
        # (e.g., SPX ~1.2-1.5%/yr from S&P 500 dividends).  If parity
        # estimation fails for all slices, fall back to the representative
        # ETF's trailing yield (approximation).  See _estimate_index_dividend_yield.
        # NOTE: q is set per-slice inside the loop below; here we set the
        # surface-level q as a fallback.
        q = 0.0  # will be updated after slice loop
    else:
        q = _get_dividend_yield(ticker)
        if q is None:
            _logger.warning(
                "Dividend yield unavailable for %s (dividendYield missing "
                "from ticker info); substituting q=0.0",
                symbol,
            )
            q = 0.0

    # get the underlying spot price
    spot = None
    try:
        info = ticker.info or {}
        spot = info.get("regularMarketPrice") or info.get("previousClose")
    except Exception:
        _logger.warning(f"Failed to fetch spot price for {symbol!r}", exc_info=True)
    if spot is None or not isinstance(spot, (int, float)):
        raise ValueError(f"Could not fetch spot price for {symbol!r}")

    spot = float(spot)

    # build slices from available expiries
    all_rejected: list[RejectionRecord] = []
    all_quality_drops: list[DropRecord] = []
    slices: list[ExpirySlice] = []
    ref_date = date.today()

    for exp_str in expiries:
        if len(slices) >= max_expiries:
            break

        T = (date.fromisoformat(exp_str) - ref_date).days / 365.0
        if T <= min_T_years:
            continue

        chain = ticker.option_chain(exp_str)

        # Apply data-quality filter to raw DataFrames before building Quotes
        if disable_quality_filter:
            calls_filtered = chain.calls
            puts_filtered = chain.puts
        else:
            calls_filtered, calls_drops = filter_option_chain(
                chain.calls, exp_str, quality_config
            )
            puts_filtered, puts_drops = filter_option_chain(
                chain.puts, exp_str, quality_config
            )
            all_quality_drops.extend(calls_drops)
            all_quality_drops.extend(puts_drops)

        quotes: list[Quote] = []

        for _, row in calls_filtered.iterrows():
            qq = _row_to_quote(row, OptionType.CALL)
            if qq is not None:
                quotes.append(qq)

        for _, row in puts_filtered.iterrows():
            qq = _row_to_quote(row, OptionType.PUT)
            if qq is not None:
                quotes.append(qq)

        if not quotes:
            continue

        # apply cleaning rules to this slice
        sl_raw = ExpirySlice(expiry_time=T, quotes=quotes)
        kept, rejected = clean_quotes(sl_raw, spot)
        all_rejected.extend(rejected)

        if not kept:
            continue

        slices.append(ExpirySlice(expiry_time=T, quotes=kept))

    # For index symbols, estimate q per-expiry via put-call parity.
    # If all slices fail estimation, fall back to representative ETF yield.
    if _is_index and slices:
        from statistics import median as _median
        per_slice_qs: list[float] = []
        parity_slices: list[float] = []
        failed_parity_slices: list[float] = []
        for sl in slices:
            q_est = _estimate_index_dividend_yield(sl, spot, r)
            if q_est is not None:
                sl.div_yield = q_est
                per_slice_qs.append(q_est)
                parity_slices.append(sl.expiry_time)
            else:
                failed_parity_slices.append(sl.expiry_time)
        if per_slice_qs:
            q = _median(per_slice_qs)
        else:
            rep_q = _get_representative_dividend_yield(symbol)
            if rep_q is not None:
                q = rep_q
            # else q stays at 0.0 from the initial assignment

        # Visibility only — no value changes.  Report which slices got a
        # genuine per-expiry parity q and which fell back to the
        # surface-level q, and where that surface q came from, so a mixed
        # q-quality surface is never silent.
        if failed_parity_slices and per_slice_qs:
            _logger.warning(
                "Index %s: q quality is MIXED across slices — %d/%d used "
                "per-expiry put-call parity q (%s); %d/%d used the "
                "surface-level q (median of parity estimates, q=%.6f) "
                "(%s)",
                symbol, len(parity_slices), len(slices), parity_slices,
                len(failed_parity_slices), len(slices), q,
                failed_parity_slices,
            )
        elif not per_slice_qs:
            if q != 0.0:
                _logger.warning(
                    "Index %s: put-call parity q failed for all %d slices; "
                    "surface q from representative ETF yield (q=%.6f)",
                    symbol, len(slices), q,
                )
            else:
                _logger.warning(
                    "Index %s: put-call parity q failed for all %d slices "
                    "and no representative ETF yield available; surface q "
                    "hardcoded to 0.0",
                    symbol, len(slices),
                )

    return (
        VolSurface(spot=spot, risk_free=r, div_yield=q, slices=slices),
        all_rejected,
        all_quality_drops,
    )
