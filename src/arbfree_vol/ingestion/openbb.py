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
from datetime import date
from typing import Any

from arbfree_vol.data.quality import (
    DataQualityConfig,
    DropRecord,
    filter_option_chain,
)
from arbfree_vol.data.snapshot_guard import check_snapshot_time
from arbfree_vol.ingestion.cleaning import RejectionRecord, clean_quotes
from arbfree_vol.ingestion._index_rates import (
    _estimate_index_dividend_yield,
    _get_representative_dividend_yield,
)
from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface

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
    ``lastPrice``.  Falls back to ``lastPrice`` (the normalized form of
    OpenBB's ``last_trade_price``; see ``_normalise_columns``) if either
    bid or ask is missing.  Returns ``None`` when no valid price can be
    determined.
    """
    bid = _safe_float(row.get("bid"), None)
    ask = _safe_float(row.get("ask"), None)
    last = _safe_float(row.get("lastPrice"), None)

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
    # Coerce numerics but PRESERVE missingness (NaN, not 0.0) for the
    # market-data columns so ``filter_option_chain`` can tell a missing
    # value apart from a genuinely observed zero.  ``strike`` keeps a
    # 0.0 default — a missing strike is unpriceable either way.
    for col in ("volume", "openInterest", "bid", "ask"):
        if col in out.columns:
            out[col] = out[col].apply(lambda x: _safe_float(x, float("nan")))
    if "strike" in out.columns:
        out["strike"] = out["strike"].apply(lambda x: _safe_float(x, 0.0))
    return out


def _expiry_to_date_str(x: Any) -> str:
    """Normalize an OpenBB expiration value to a date-only ISO string.

    OpenBB providers hand back expirations as ``date``, ``datetime``,
    ``pandas.Timestamp`` or ISO-format strings — sometimes carrying a
    time component (e.g. ``2026-08-11T00:00:00``).  The sort/parse path
    in ``fetch_chain`` feeds these straight into ``date.fromisoformat``,
    which only accepts date-only strings, so the time component must be
    stripped here.
    """
    to_date = getattr(x, "date", None)
    if callable(to_date):
        # datetime / pandas.Timestamp expose a callable .date() method.
        # Plain ``date`` objects do NOT (``getattr`` returns None), so
        # they fall through to the string branch below instead.
        return to_date().isoformat()
    # String form (or a bare ``date`` object): strip any trailing time
    # component ("T" or space).
    return str(x).strip().split("T")[0].split(" ")[0]


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
            _logger.warning(
                "OpenBB equity price quote failed for %s; spot left "
                "undetermined",
                symbol, exc_info=True,
            )
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
        # The q=0.0 here is a PLACEHOLDER, not an observation — index
        # symbols estimate q per-expiry via put-call parity after the
        # slice loop.  It must never be logged as an observed zero (the
        # pre-fix code hit the `q == 0.0` observed-zero branch for every
        # index symbol because the placeholder triggered it).
        _logger.warning(
            "Dividend yield for %s starts at the index default q=0.0 "
            "(placeholder); per-expiry put-call parity estimation runs "
            "after the slice loop",
            symbol,
        )
    else:
        try:
            yf_ticker = yf.Ticker(symbol)
            info = yf_ticker.info or {}
            div = info.get("dividendYield")
            if div is not None and isinstance(div, (int, float)):
                q = float(div)
                if math.isnan(q):
                    q = None
                elif q > 0.50:
                    q /= 100.0
        except Exception:
            _logger.warning("Failed to fetch dividend yield", exc_info=True)
        if q is None:
            _logger.warning(
                "Dividend yield unavailable for %s (dividendYield missing "
                "from ticker info); substituting q=0.0",
                symbol,
            )
            q = 0.0
        elif q == 0.0:
            # An observed zero is a real observation, not a substitution:
            # the value is used as-is, but the provenance is logged so a
            # zero-yield surface is never silent about where q came from.
            _logger.warning(
                "Dividend yield for %s observed as zero (dividendYield "
                "present as 0.0 in ticker info); using q=0.0 as observed",
                symbol,
            )

    if r is None:
        _logger.warning(
            "Risk-free rate unavailable for %s (^IRX fetch failed or "
            "empty); substituting r=0.05",
            symbol,
        )
        r = 0.05

    # ── Normalise columns ────────────────────────────────────────────
    df = _normalise_columns(raw_df)

    # ── Group by expiry and build slices ─────────────────────────────
    all_rejected: list[RejectionRecord] = []
    all_quality_drops: list[DropRecord] = []
    slices: list[ExpirySlice] = []
    ref_date = date.today()

    # OpenBB 'expiration' column may be date/datetime/Timestamp or a
    # string (sometimes carrying a time component).  Normalize to a
    # date-only string so the sort and ``date.fromisoformat`` parse path
    # cannot be broken by a time component.
    if "expiration" in df.columns:
        df["_expiry_str"] = df["expiration"].apply(_expiry_to_date_str)
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
            elif rep_q is not None:
                # The representative yield was PRESENT and observed as
                # zero (the helper itself logs the observed-zero
                # provenance).  This branch must not claim the yield was
                # unavailable — an observed zero is an observation, not
                # a substitution.
                _logger.warning(
                    "Index %s: put-call parity q failed for all %d slices; "
                    "representative ETF yield observed as zero; surface "
                    "q=0.0 as observed",
                    symbol, len(slices),
                )
            else:
                _logger.warning(
                    "Index %s: put-call parity q failed for all %d slices "
                    "and no representative ETF yield available; surface q "
                    "hardcoded to 0.0",
                    symbol, len(slices),
                )

    if not slices:
        raise ValueError(
            f"No valid slices produced for {symbol!r} — check data availability"
        )

    return (
        VolSurface(spot=spot, risk_free=r, div_yield=q, slices=slices),
        all_rejected,
        all_quality_drops,
    )
