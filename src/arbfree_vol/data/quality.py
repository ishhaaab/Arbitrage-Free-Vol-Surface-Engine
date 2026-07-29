"""Data quality filter for raw option chain DataFrames.

Applies market microstructure thresholds (minimum open interest, minimum
volume, maximum bid-ask spread) to raw yfinance option chain DataFrames
before building MarketSlices.  This is a pre-ingestion filter that catches
thinly-traded or no-quote strikes that could degrade calibration quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True, slots=True)
class DataQualityConfig:
    """Thresholds for the data-quality filter.

    All thresholds are applied independently — a strike that fails
    *any* threshold is dropped.
    """
    min_open_interest: int = 10
    min_volume: int = 0
    max_bid_ask_pct: float = 50.0  # percentage, not fraction


@dataclass(frozen=True, slots=True)
class DropRecord:
    """Audit record for a strike that was dropped by the data-quality filter."""
    strike: float
    expiry: str
    reason: str
    open_interest: int
    volume: int
    bid_ask_pct: float


def filter_option_chain(
    df: pd.DataFrame,
    expiry: str,
    config: DataQualityConfig | None = None,
) -> tuple[pd.DataFrame, list[DropRecord]]:
    """Filter a raw yfinance option chain DataFrame by data quality thresholds.

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
    """
    import math

    if config is None:
        config = DataQualityConfig()

    drops: list[DropRecord] = []
    keep_mask = pd.Series(True, index=df.index)

    for idx, row in df.iterrows():
        strike = float(row.get("strike", 0) or 0)

        raw_oi = row.get("openInterest", 0)
        oi = int(raw_oi) if raw_oi is not None and not (isinstance(raw_oi, float) and math.isnan(raw_oi)) else 0

        raw_vol = row.get("volume", 0)
        vol = int(raw_vol) if raw_vol is not None and not (isinstance(raw_vol, float) and math.isnan(raw_vol)) else 0

        raw_bid = row.get("bid", 0)
        bid = float(raw_bid) if raw_bid is not None and not (isinstance(raw_bid, float) and math.isnan(raw_bid)) else 0.0

        raw_ask = row.get("ask", 0)
        ask = float(raw_ask) if raw_ask is not None and not (isinstance(raw_ask, float) and math.isnan(raw_ask)) else 0.0

        # Compute bid-ask spread as % of mid
        mid = (bid + ask) / 2.0
        if mid > 0:
            bid_ask_pct = (ask - bid) / mid * 100.0
        else:
            bid_ask_pct = 0.0  # no quote — will be caught by zero_bid_ask

        # Check thresholds
        reason_parts: list[str] = []

        if oi < config.min_open_interest:
            reason_parts.append(f"OI={oi}<{config.min_open_interest}")

        if vol < config.min_volume:
            reason_parts.append(f"vol={vol}<{config.min_volume}")

        if mid > 0 and bid_ask_pct > config.max_bid_ask_pct:
            reason_parts.append(
                f"spread={bid_ask_pct:.1f}%>{config.max_bid_ask_pct}%"
            )

        if reason_parts:
            keep_mask.at[idx] = False
            drops.append(DropRecord(
                strike=strike,
                expiry=expiry,
                reason="; ".join(reason_parts),
                open_interest=oi,
                volume=vol,
                bid_ask_pct=bid_ask_pct,
            ))

    return df[keep_mask].copy(), drops
