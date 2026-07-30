"""Fetch live option chains from yfinance and build a VolSurface.

Attempts to source real risk-free rates and dividend yields.  Falls
back to pre-pass forward-curve estimation when rates are unavailable.
"""

import logging
import math
from datetime import date
from typing import Any

import yfinance as yf

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.repair.fwd_curve import estimate_forward_curve
from arbfree_vol.ingestion.cleaning import clean_quotes, RejectionRecord
from arbfree_vol.data.quality import (
    DataQualityConfig,
    DropRecord,
    filter_option_chain,
)

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
    failing any threshold (min open interest, min volume, max bid-ask
    spread) are dropped and recorded in the returned ``quality_drops``
    list.

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
    ticker = yf.Ticker(symbol)
    expiries = ticker.options

    if not expiries:
        raise ValueError(f"No expiries available for symbol {symbol!r}")

    # source rates
    r = _get_risk_free_rate()
    q = _get_dividend_yield(ticker)
    if r is None or q is None:
        # fallback: detect_with_forward() will correct via pre-pass
        r = r or 0.05
        q = q or 0.0

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

    return (
        VolSurface(spot=spot, risk_free=r, div_yield=q, slices=slices),
        all_rejected,
        all_quality_drops,
    )
