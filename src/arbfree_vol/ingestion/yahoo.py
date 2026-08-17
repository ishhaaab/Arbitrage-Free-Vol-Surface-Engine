"""Fetch live option chains from yfinance and build a VolSurface.

This module is named ``yahoo`` (not ``yfinance``) so it does not
shadow the third-party ``yfinance`` package it imports internally.

Attempts to source real risk-free rates and dividend yields.  Falls
back to pre-pass forward-curve estimation when rates are unavailable.

Index symbols (tickers starting with ``^``, e.g. ``^SPX``) use a
put-call-parity implied dividend yield per expiry (via
``_estimate_index_dividend_yield``), with a representative-ETF
fallback (``_INDEX_REPRESENTATIVE``, e.g. ``^SPX`` -> SPY) when parity
estimation fails.  Per-slice q choices are logged.
"""

import logging
import warnings
from datetime import date

import yfinance as yf

from arbfree_vol.models.surface import VolSurface, ExpirySlice
from arbfree_vol.ingestion.cleaning import RejectionRecord
from arbfree_vol.ingestion._common import build_slice
from arbfree_vol.ingestion._index_rates import (
    estimate_index_dividend_yields,
    fetch_rates,
)
from arbfree_vol.data.quality import DataQualityConfig, DropRecord
from arbfree_vol.data.snapshot_guard import check_snapshot_time

_logger = logging.getLogger(__name__)


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

    # source rates (shared orchestration: ^IRX r, per-symbol q, and the
    # documented fallbacks; reuses the already-created ticker for q)
    _is_index = symbol.startswith("^")
    r, q = fetch_rates(symbol, _is_index, ticker)

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

        sl, rejected, drops = build_slice(
            chain.calls, chain.puts, exp_str, T, spot,
            quality_config, disable_quality_filter,
        )
        all_quality_drops.extend(drops)
        all_rejected.extend(rejected)
        if sl is not None:
            slices.append(sl)

    # For index symbols, estimate q per-expiry via put-call parity.
    # If all slices fail estimation, fall back to representative ETF yield.
    if _is_index and slices:
        q = estimate_index_dividend_yields(slices, spot, r, symbol)

    return (
        VolSurface(spot=spot, risk_free=r, div_yield=q, slices=slices),
        all_rejected,
        all_quality_drops,
    )
