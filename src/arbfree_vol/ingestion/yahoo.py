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
    apply_curve_rates,
    fetch_rates,
    resolve_index_q,
    resolve_rate_curve,
)
from arbfree_vol.data.quality import DataQualityConfig, DropRecord
from arbfree_vol.data.snapshot_guard import check_snapshot_time
from arbfree_vol.rates import YieldTermStructure
from arbfree_vol.time import DayCount, Calendar

_logger = logging.getLogger(__name__)


def _resolve_rates(
    symbol: str,
    ticker: yf.Ticker,
    *,
    curve: YieldTermStructure | None = None,
    use_fred_curve: bool = False,
) -> tuple[float, float, YieldTermStructure | None]:
    """Rate seam: supplied curve > FRED curve > shared ^IRX orchestration.

    Returns ``(r, q, fred_curve)``.  ``fred_curve`` is ``None`` on the
    ^IRX path; when a curve is present, surface-level ``r`` is its rate
    at ~1y for display and per-slice ``r(T)`` is threaded later by
    ``apply_curve_rates``.  ``q`` comes from the shared orchestration
    either way.
    """
    is_index = symbol.startswith("^")
    fred_curve = resolve_rate_curve(curve, use_fred_curve)
    if fred_curve is not None:
        _, q = fetch_rates(symbol, is_index, ticker)
        r = fred_curve.zero_rate(1.0)
    else:
        r, q = fetch_rates(symbol, is_index, ticker)
    return r, q, fred_curve


def _fetch_spot(ticker: yf.Ticker, symbol: str) -> float:
    """Spot seam: ``info.regularMarketPrice`` or ``previousClose``.

    Raises ``ValueError`` with a clear message when the ticker yields no
    usable price (so the failure mode is a named, visible one).
    """
    spot = None
    try:
        info = ticker.info or {}
        spot = info.get("regularMarketPrice") or info.get("previousClose")
    except Exception:
        _logger.warning(f"Failed to fetch spot price for {symbol!r}", exc_info=True)
    if spot is None or not isinstance(spot, (int, float)):
        raise ValueError(f"Could not fetch spot price for {symbol!r}")
    return float(spot)


def _build_slices(
    ticker: yf.Ticker,
    expiries: list[str],
    *,
    max_expiries: int,
    min_T_years: float,
    spot: float,
    ref_date: date,
    dc: DayCount,
    cal: Calendar | None,
    quality_config: DataQualityConfig | None,
    disable_quality_filter: bool,
) -> tuple[list[ExpirySlice], list[RejectionRecord], list[DropRecord]]:
    """Expiry seam: per-expiry day-count/calendar ``T``, then ``build_slice``.

    Walks the nearest expiries up to ``max_expiries``, skipping any
    whose maturity is at or below ``min_T_years``, and returns the
    slices plus the rejection / quality-drop audit trails.
    """
    all_rejected: list[RejectionRecord] = []
    all_quality_drops: list[DropRecord] = []
    slices: list[ExpirySlice] = []

    for exp_str in expiries:
        if len(slices) >= max_expiries:
            break

        exp_date = date.fromisoformat(exp_str)
        if cal is not None and not cal.is_business_day(exp_date):
            exp_date = cal.adjust(exp_date, "following")
        T = dc.year_fraction(ref_date, exp_date)
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

    return slices, all_rejected, all_quality_drops


def fetch_chain(
    symbol: str,
    max_expiries: int = 5,
    min_T_years: float = 7.0 / 365.0,
    quality_config: DataQualityConfig | None = None,
    disable_quality_filter: bool = False,
    curve: YieldTermStructure | None = None,
    day_count: DayCount | str = "ACT/365F",
    calendar: Calendar | str | None = None,
    use_fred_curve: bool = False,
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

    # day-count / calendar
    _dc = DayCount(day_count) if isinstance(day_count, str) else day_count
    _cal: Calendar | None = None
    if calendar is not None:
        _cal = Calendar(calendar) if isinstance(calendar, str) else calendar  # type: ignore[arg-type]

    # source rates: supplied curve > FRED curve > ^IRX orchestration
    r, q, _fred_curve = _resolve_rates(
        symbol, ticker, curve=curve, use_fred_curve=use_fred_curve
    )

    # get the underlying spot price
    spot = _fetch_spot(ticker, symbol)

    # build slices from available expiries
    ref_date = date.today()
    slices, all_rejected, all_quality_drops = _build_slices(
        ticker,
        expiries,
        max_expiries=max_expiries,
        min_T_years=min_T_years,
        spot=spot,
        ref_date=ref_date,
        dc=_dc,
        cal=_cal,
        quality_config=quality_config,
        disable_quality_filter=disable_quality_filter,
    )

    # per-slice r(T) from curve when available
    apply_curve_rates(slices, _fred_curve)

    # Reconcile index q per-expiry via put-call parity.  Non-index
    # symbols (or an empty chain) keep the pre-loop q unchanged; if all
    # slices fail estimation, the seam falls back to representative ETF
    # yield.
    q = resolve_index_q(slices, spot, r, symbol, q)

    return (
        VolSurface(spot=spot, risk_free=r, div_yield=q, slices=slices),
        all_rejected,
        all_quality_drops,
    )
