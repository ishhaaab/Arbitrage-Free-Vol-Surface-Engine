"""Fetch live option chains from OpenBB and build a VolSurface.

OpenBB wraps multiple data providers behind a unified API.  This module
uses the ``yfinance`` provider by default (same underlying data as
``ingestion.yahoo`` but via OpenBB's normalised column schema), and
falls back to other free providers if available.

Requires ``openbb`` — install with ``pip install openbb``.
"""

from __future__ import annotations

import logging
import math
import warnings
from datetime import date
from typing import Any

from arbfree_vol.data.quality import DataQualityConfig, DropRecord
from arbfree_vol.data.snapshot_guard import check_snapshot_time
from arbfree_vol.ingestion.cleaning import RejectionRecord
from arbfree_vol.ingestion._common import build_slice
from arbfree_vol.ingestion._index_rates import (
    estimate_index_dividend_yields,
    fetch_rates,
)
from arbfree_vol.models.surface import ExpirySlice, VolSurface

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


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert a value to float, handling None/NaN."""
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return float(val)


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

def _fetch_spot(obb, symbol: str, provider: str, raw_df) -> float | None:
    """Determine the spot price from the chain or an OpenBB equity quote.

    Prefers the chain's ``underlying_price`` column; falls back to an
    OpenBB equity price quote when the chain lacks it.
    """
    if "underlying_price" in raw_df.columns:
        spot = _safe_float(raw_df.get("underlying_price").iloc[0], None)
        if spot is not None:
            return spot
    # Try to get spot from OpenBB equity price
    try:
        quote = obb.equity.price.quote(symbol, provider=provider)
        quote_df = quote.to_df()
        if not quote_df.empty:
            return _safe_float(quote_df.iloc[0].get("last_price"), None)
    except Exception:
        _logger.warning(
            "OpenBB equity price quote failed for %s; spot left "
            "undetermined",
            symbol, exc_info=True,
        )
    return None


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
        data as ``ingestion.yahoo``).  Other free providers may work
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
    spot = _fetch_spot(obb, symbol, provider, raw_df)
    if spot is None or spot <= 0:
        raise ValueError(f"Could not determine spot price for {symbol!r}")
    spot = float(spot)

    # ── Risk-free rate and dividend yield ────────────────────────────
    _is_index = symbol.startswith("^")
    r, q = fetch_rates(symbol, _is_index)

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

        sl, rejected, drops = _build_expiry_slice(
            df, exp_str, T, spot,
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

    if not slices:
        raise ValueError(
            f"No valid slices produced for {symbol!r} — check data availability"
        )

    return (
        VolSurface(spot=spot, risk_free=r, div_yield=q, slices=slices),
        all_rejected,
        all_quality_drops,
    )


def _build_expiry_slice(
    df,
    exp_str: str,
    T: float,
    spot: float,
    quality_config: DataQualityConfig | None,
    disable_quality_filter: bool,
) -> tuple[ExpirySlice | None, list[RejectionRecord], list[DropRecord]]:
    """Build one expiry's slice from the normalised chain DataFrame.

    Splits the expiry's rows into calls and puts, then delegates to the
    shared ``build_slice``.  Returns ``(slice_or_None, rejected, drops)``.
    """
    exp_df = df[df["_expiry_str"] == exp_str].copy()

    # Split calls and puts
    calls_raw = exp_df[exp_df["option_type"].str.lower().isin(["call", "c"])].copy()
    puts_raw = exp_df[exp_df["option_type"].str.lower().isin(["put", "p"])].copy()

    return build_slice(
        calls_raw, puts_raw, exp_str, T, spot,
        quality_config, disable_quality_filter,
    )
