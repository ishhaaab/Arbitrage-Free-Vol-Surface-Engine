"""Data quality filter for raw option chain DataFrames.

Applies market microstructure thresholds (minimum open interest, maximum
bid-ask spread) to raw option chain DataFrames before building MarketSlices.
This is a pre-ingestion filter that catches thinly-traded or no-quote strikes
that could degrade calibration quality.

Volume is **not** used as a filter criterion — daily per-strike volume=0 is
normal for legitimate market-maker quotes away from ATM, and is not a reliable
per-strike liquidity signal (unlike open interest or bid-ask width).  Filtering
on volume would drop good strikes, not just bad ones.  Volume is still
recorded in ``DropRecord`` for diagnostic context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class DataQualityConfig:
    """Thresholds for the data-quality filter.

    Only ``min_open_interest`` and ``max_bid_ask_pct`` are enforced.
    Volume is intentionally excluded: daily per-strike volume=0 is normal
    for legitimate market-maker quotes away from ATM and is not a reliable
    per-strike liquidity signal (unlike open interest or bid-ask width).
    Filtering on volume would drop good strikes, not just bad ones.

    All thresholds are applied independently — a strike that fails
    *any* threshold is dropped.
    """
    min_open_interest: int = 10
    max_bid_ask_pct: float = 50.0  # percentage, not fraction


@dataclass(frozen=True, slots=True)
class DropRecord:
    """Audit record for a strike that was dropped by the data-quality filter.

    ``volume`` is included for diagnostic context only — it is not a
    pass/fail criterion (volume is never compared against a threshold).

    ``missing_fields`` lists which market-data fields were absent
    (missing / NaN / pd.NA) rather than genuinely observed as zero —
    e.g. ``("open_interest",)`` when the provider returned no OI value.
    This keeps a missing value distinguishable from a real zero, so a
    mass drop caused by a provider omitting a column is never
    mislabelled as "illiquid strike".
    """
    strike: float
    expiry: str
    reason: str
    open_interest: int
    volume: int
    bid_ask_pct: float
    missing_fields: tuple[str, ...] = ()


def _is_missing(value: Any) -> bool:
    """True when a market-data value is absent (None / NaN / pd.NA).

    A missing value is NOT the same as an observed zero — the caller
    must be able to tell the two apart.
    """
    import math

    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _build_keep_mask(
    config: DataQualityConfig,
    df: pd.DataFrame,
    expiry: str,
) -> tuple[pd.Series, list[DropRecord]]:
    """Row-by-row data-quality filter for one option-chain DataFrame.

    Returns the keep mask and the accumulated DropRecords.
    """
    keep_mask = pd.Series(True, index=df.index)
    drops: list[DropRecord] = []
    for idx, row in df.iterrows():
        strike = float(row.get("strike", 0) or 0)

        keep, drop = _evaluate_row(config, row, strike, expiry)
        if not keep:
            keep_mask.at[idx] = False
            drops.append(drop)
    return keep_mask, drops


def _evaluate_row(
    config: DataQualityConfig,
    row,
    strike: float,
    expiry: str,
) -> tuple[bool, DropRecord | None]:
    """Evaluate one chain row against the data-quality thresholds.

    Returns ``(keep, drop)`` — ``keep`` is True when the row survives,
    otherwise ``drop`` is the audit record describing why.
    """
    missing: list[str] = []
    oi = _extract_int_field(row, "openInterest", "open_interest", missing)
    vol = _extract_int_field(row, "volume", "volume", missing)
    bid = _extract_float_field(row, "bid", "bid", missing)
    ask = _extract_float_field(row, "ask", "ask", missing)

    # Compute bid-ask spread as % of mid
    mid = (bid + ask) / 2.0
    if mid > 0:
        bid_ask_pct = (ask - bid) / mid * 100.0
    else:
        bid_ask_pct = 0.0  # only meaningful when both sides are present

    # Check thresholds
    reason_parts: list[str] = []
    missing_sides = [side for side in ("bid", "ask") if side in missing]

    if oi < config.min_open_interest:
        if "open_interest" in missing:
            reason_parts.append(f"OI=missing<{config.min_open_interest}")
        else:
            reason_parts.append(f"OI={oi}<{config.min_open_interest}")

    if len(missing_sides) >= 1:
        # One- or two-sided missing quote: the mid would be fabricated
        # from an available side (missing bid → +200%, missing ask →
        # −200%) or is unknowable (both missing → 0), so the true
        # spread cannot be computed.  Flag the row instead of passing
        # it with a made-up mid — same missing-vs-observed-zero class
        # as open interest.  (The both-missing case previously fell
        # through: mid=0 skipped this branch and the no-quote strike
        # passed the filter — the N1 no-quote path.)
        reason_parts.append(
            f"spread=missing (missing: {', '.join(missing_sides)})"
        )
    elif bid_ask_pct > config.max_bid_ask_pct:
        reason_parts.append(
            f"spread={bid_ask_pct:.1f}%>{config.max_bid_ask_pct}%"
        )

    if reason_parts:
        return False, DropRecord(
            strike=strike,
            expiry=expiry,
            reason="; ".join(reason_parts),
            open_interest=oi,
            volume=vol,
            bid_ask_pct=bid_ask_pct,
            missing_fields=tuple(missing),
        )
    return True, None


def _extract_int_field(row, key: str, label: str, missing: list[str]) -> int:
    """Extract an integer market-data field, tracking missing values.

    ``label`` is the display name recorded in ``missing`` (e.g. column
    ``openInterest`` → label ``open_interest``).  No zero default: an
    ABSENT column (``row.get(key)`` returns None) is a missing value,
    exactly like None/NaN/pd.NA in a present column — never an observed
    zero (the old ``row.get(key, 0)`` conflated the two).
    """
    raw = row.get(key)
    if _is_missing(raw):
        missing.append(label)
        return 0
    return int(raw)


def _extract_float_field(row, key: str, label: str, missing: list[str]) -> float:
    """Extract a float market-data field, tracking missing values."""
    raw = row.get(key)
    if _is_missing(raw):
        missing.append(label)
        return 0.0
    return float(raw)


def filter_option_chain(
    df: pd.DataFrame,
    expiry: str,
    config: DataQualityConfig | None = None,
) -> tuple[pd.DataFrame, list[DropRecord]]:
    """Filter a raw option chain DataFrame by data quality thresholds.

    Only two thresholds are enforced: ``min_open_interest`` and
    ``max_bid_ask_pct``.  Volume is recorded in ``DropRecord`` for
    diagnostic context but is not a pass/fail criterion.

    Parameters
    ----------
    df:
        Raw DataFrame from ``yfinance.Ticker.option_chain()`` — either
        ``calls`` or ``puts``.  Expected columns: ``strike``,
        ``openInterest``, ``volume``, ``bid``, ``ask``.
    expiry:
        Expiry date string (e.g. ``"2026-08-15"``) for audit records.
    config:
        Thresholds.  Uses ``DataQualityConfig()`` defaults if ``None``.

    Returns
    -------
    (filtered_df, drops)
        ``filtered_df`` is the surviving rows; ``drops`` is the list of
        ``DropRecord`` entries for rows that failed a threshold.

    Missing values are NOT silently treated as zero: a row whose
    ``openInterest`` is absent (``None``/NaN/pd.NA) is dropped with
    reason ``OI=missing<...`` and ``missing_fields=("open_interest",)``
    so it stays distinguishable from a genuinely observed ``OI=0``.
    The same applies to any quote with one or both sides missing
    (``bid`` and/or ``ask`` absent): it is dropped with reason
    ``spread=missing (missing: <side[, side]>)`` instead of passing
    with a mid fabricated from the available side — a no-quote strike
    (both sides missing) is the filter's documented target.  Missing
    ``volume`` is recorded in ``missing_fields`` only — volume is
    never a criterion.

    An absent COLUMN (the provider omitted the field entirely) is
    treated the same as a missing value: the market-data fields are read
    with ``row.get(key)`` and NO zero default, so a DataFrame without an
    ``openInterest`` column flags every row's OI as missing rather than
    as an observed zero.  ``strike`` keeps its 0.0 default — a row with
    no strike is unpriceable either way.
    """
    if config is None:
        config = DataQualityConfig()

    keep_mask, drops = _build_keep_mask(config, df, expiry)

    return df[keep_mask].copy(), drops
