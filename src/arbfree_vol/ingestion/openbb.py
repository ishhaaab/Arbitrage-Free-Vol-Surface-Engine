"""Fetch live option chains from OpenBB and build a VolSurface.

OpenBB wraps multiple data providers behind a unified API.  This module
uses the ``yfinance`` provider by default (same underlying data as
``ingestion.yfinance`` but via OpenBB's normalised column schema), and
falls back to other free providers if available.

Requires ``openbb`` — install with ``pip install openbb``.
"""

from __future__ import annotations

import logging
import math
import warnings
from datetime import date, datetime, time
from typing import Any

from arbfree_vol.data.quality import (
    DataQualityConfig,
    DropRecord,
    filter_option_chain,
)
from arbfree_vol.data.snapshot_guard import check_snapshot_time
from arbfree_vol.ingestion.cleaning import RejectionRecord, clean_quotes
from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface
from arbfree_vol.repair.fwd_curve import estimate_forward_curve

_logger = logging.getLogger(__name__)


# ── Column mapping ───────────────────────────────────────────────────
# OpenBB (yfinance provider) uses different column names from raw yfinance.
#   OpenBB               yfinance
#   ─────────────────    ───────────────
#   open_interest        openInterest
#   last_trade_price     lastPrice
#   implied_volatility   impliedVolatility
#   option_type          (type)  values: 'call'/'put'
#   expiration           (expiry date)

_OPTION_TYPE_MAP = {
    "call": OptionType.CALL,
    "put": OptionType.PUT,
    "C": OptionType.CALL,
    "P": OptionType.PUT,
}


def _safe_int(val: Any, default: int = 0) -> int:
    """Convert a value to int, handling None/NaN."""
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return int(val)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert a value to float, handling None/NaN."""
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return float(val)


def _row_to_quote(row: Any, otype: OptionType) -> Quote | None:
    """Convert an OpenBB DataFrame row to a Quote.

    Uses the **mid price** (``(bid + ask) / 2``) when both bid and ask
    are available — this reflects the live market, not stale
    ``last_trade_price``.  Falls back to ``last_trade_price`` if either
    bid or ask is missing.  Returns ``None`` when no valid price can be
    determined.
    """
    bid = _safe_float(row.get("bid"), None)
    ask = _safe_float(row.get("ask"), None)
    last = _safe_float(row.get("last_trade_price"), None)

    if bid is not None and ask is not None:
        price = (bid + ask) / 2.0
    elif last is not None:
        price = last
        bid = None
        ask = None
    else:
        return None

    return Quote(
        strike=float(row["strike"]),
        option_type=otype,
        price=price,
        bid=bid,
        ask=ask,
    )


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
    import yfinance as yf_local

    rep = _INDEX_REPRESENTATIVE.get(symbol)
    if rep is None:
        return None
    try:
        rep_ticker = yf_local.Ticker(rep)
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


def _normalise_columns(df):
    """Rename OpenBB columns to match the yfinance schema expected by
    ``filter_option_chain`` and other pipeline components.

    Returns a new DataFrame — does not mutate the input.
    """
    rename_map = {
        "open_interest": "openInterest",
        "last_trade_price": "lastPrice",
        "implied_volatility": "impliedVolatility",
    }
    out = df.rename(columns=rename_map)
    # Ensure volume and openInterest are numeric (some providers return objects)
    for col in ("volume", "openInterest", "bid", "ask", "strike"):
        if col in out.columns:
            out[col] = out[col].apply(lambda x: _safe_float(x, 0.0))
    return out


# ── Main entry point ─────────────────────────────────────────────────

def fetch_chain(
    symbol: str,
    max_expiries: int = 8,
    min_T_years: float = 0.02,
    quality_config: DataQualityConfig | None = None,
    disable_quality_filter: bool = False,
    provider: str = "yfinance",
) -> tuple[VolSurface, list[RejectionRecord], list[DropRecord]]:
    """Fetch an option chain from OpenBB and return a cleaned VolSurface.

    Steps through the available expiries, builds quotes from mid prices,
    applies the cleaning layer (wide spreads, crossed markets, deep
    moneyness, near-expiry, zero prices), and returns the cleaned surface
    plus an audit trail of rejects.

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
        raw OpenBB data.  This is the ONLY way to get truly unfiltered
        data — passing ``quality_config=None`` with
        ``disable_quality_filter=False`` still applies default thresholds.
    provider:
        OpenBB data provider.  Default ``"yfinance"`` (same underlying
        data as ``ingestion.yfinance``).  Other free providers may work
        without API keys (e.g. ``"cboe"``).

    Returns
    -------
    (surface, rejected, quality_drops)

    Raises
    ------
    ImportError
        If the ``openbb`` package is not installed.
    ValueError
        If no expiries are available or the spot price cannot be fetched.
    """
    try:
        from openbb import obb
    except ImportError:
        raise ImportError(
            "The 'openbb' package is required for this data source. "
            "Install it with:  pip install openbb"
        )

    # Snapshot-time guard (warns, does not block)
    guard_warning = check_snapshot_time()
    if guard_warning:
        warnings.warn(guard_warning, stacklevel=2)

    # ── Fetch the full chain ─────────────────────────────────────────
    _logger.info("Fetching %s options via OpenBB (provider=%s)", symbol, provider)
    try:
        result = obb.derivatives.options.chains(symbol, provider=provider)
    except Exception as exc:
        raise ValueError(
            f"Failed to fetch option chains for {symbol!r} via OpenBB "
            f"(provider={provider!r}): {exc}"
        ) from exc

    raw_df = result.to_df()
    if raw_df.empty:
        raise ValueError(f"No option chain data returned for {symbol!r}")

    # ── Spot price ───────────────────────────────────────────────────
    spot = _safe_float(raw_df.get("underlying_price").iloc[0] if "underlying_price" in raw_df.columns else None, None)
    if spot is None:
        # Try to get spot from OpenBB equity price
        try:
            quote = obb.equity.price.quote(symbol, provider=provider)
            quote_df = quote.to_df()
            if not quote_df.empty:
                spot = _safe_float(quote_df.iloc[0].get("last_price"), None)
        except Exception:
            pass
    if spot is None or spot <= 0:
        raise ValueError(f"Could not determine spot price for {symbol!r}")
    spot = float(spot)

    # ── Risk-free rate and dividend yield ────────────────────────────
    import yfinance as yf
    r = None
    q = None
    try:
        irx = yf.Ticker("^IRX")
        info = irx.info or {}
        rate = info.get("regularMarketPrice") or info.get("previousClose")
        if rate is not None and isinstance(rate, (int, float)) and rate > 0:
            r = rate / 100.0
    except Exception:
        _logger.warning("Failed to fetch risk-free rate from ^IRX", exc_info=True)

    # Index symbols (^SPX, ^VIX, etc.): estimate q per-expiry via put-call
    # parity rather than hardcoding q=0.  Indices have a genuine implied
    # dividend yield from their constituents (e.g., SPX ~1.2-1.5%/yr).
    # Per-slice estimation is done after the slice-building loop below.
    _is_index = symbol.startswith("^")
    if _is_index:
        q = 0.0  # will be updated after slice loop
    else:
        try:
            yf_ticker = yf.Ticker(symbol)
            info = yf_ticker.info or {}
            div = info.get("dividendYield")
            if div is not None and isinstance(div, (int, float)) and div > 0:
                q = float(div)
                if q > 0.50:
                    q /= 100.0
        except Exception:
            _logger.warning("Failed to fetch dividend yield", exc_info=True)

    r = r or 0.05
    q = q or 0.0

    # ── Normalise columns ────────────────────────────────────────────
    df = _normalise_columns(raw_df)

    # ── Group by expiry and build slices ─────────────────────────────
    all_rejected: list[RejectionRecord] = []
    all_quality_drops: list[DropRecord] = []
    slices: list[ExpirySlice] = []
    ref_date = date.today()

    # OpenBB 'expiration' column may be datetime.date or string
    if "expiration" in df.columns:
        # Convert to string for uniformity
        df["_expiry_str"] = df["expiration"].apply(
            lambda x: x.isoformat() if hasattr(x, "isoformat") else str(x)
        )
    else:
        raise ValueError("OpenBB chain DataFrame missing 'expiration' column")

    expiries_sorted = sorted(df["_expiry_str"].unique())

    for exp_str in expiries_sorted:
        if len(slices) >= max_expiries:
            break

        T = (date.fromisoformat(exp_str) - ref_date).days / 365.0
        if T <= min_T_years:
            continue

        exp_df = df[df["_expiry_str"] == exp_str].copy()

        # Split calls and puts
        calls_raw = exp_df[exp_df["option_type"].str.lower().isin(["call", "c"])].copy()
        puts_raw = exp_df[exp_df["option_type"].str.lower().isin(["put", "p"])].copy()

        # Apply data-quality filter to raw DataFrames before building Quotes
        if disable_quality_filter:
            calls_filtered = calls_raw
            puts_filtered = puts_raw
        else:
            calls_filtered, calls_drops = filter_option_chain(
                calls_raw, exp_str, quality_config
            )
            puts_filtered, puts_drops = filter_option_chain(
                puts_raw, exp_str, quality_config
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
        for sl in slices:
            q_est = _estimate_index_dividend_yield(sl, spot, r)
            if q_est is not None:
                sl.div_yield = q_est
                per_slice_qs.append(q_est)
        if per_slice_qs:
            q = _median(per_slice_qs)
        else:
            rep_q = _get_representative_dividend_yield(symbol)
            if rep_q is not None:
                q = rep_q
            # else q stays at 0.0 from the initial assignment

    if not slices:
        raise ValueError(
            f"No valid slices produced for {symbol!r} — check data availability"
        )

    return (
        VolSurface(spot=spot, risk_free=r, div_yield=q, slices=slices),
        all_rejected,
        all_quality_drops,
    )
