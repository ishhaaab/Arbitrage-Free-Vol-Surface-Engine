"""Shared option-chain ingestion helpers.

``row_to_quote`` and ``build_slice`` are used by both the yfinance and
OpenBB ingestion modules.  They operate on the normalized column schema
(``strike``, ``bid``, ``ask``, ``lastPrice``) that OpenBB's
``_normalise_columns`` produces from its raw provider columns, so a single
implementation serves both sources.
"""

from __future__ import annotations

import math
from typing import Any

from arbfree_vol.data.quality import (
    DataQualityConfig,
    DropRecord,
    filter_option_chain,
)
from arbfree_vol.ingestion.cleaning import RejectionRecord, clean_quotes
from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote


def row_to_quote(row: Any, otype: OptionType) -> Quote | None:
    """Convert a DataFrame row to a Quote using mid or last price.

    Uses the mid price (``(bid + ask) / 2``) when both bid and ask are
    present; falls back to ``lastPrice`` otherwise.  Returns ``None`` when
    no valid price can be determined.
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


def build_slice(
    calls_df,
    puts_df,
    exp_str: str,
    T: float,
    spot: float,
    quality_config: DataQualityConfig | None,
    disable_quality_filter: bool,
) -> tuple[ExpirySlice | None, list[RejectionRecord], list[DropRecord]]:
    """Quality-filter, build quotes, and clean one expiry into a slice.

    Returns ``(slice_or_None, rejected, drops)``.  ``slice_or_None`` is
    ``None`` when no quotes survive cleaning.
    """
    drops: list[DropRecord] = []
    if disable_quality_filter:
        calls_filtered = calls_df
        puts_filtered = puts_df
    else:
        calls_filtered, calls_drops = filter_option_chain(calls_df, exp_str, quality_config)
        puts_filtered, puts_drops = filter_option_chain(puts_df, exp_str, quality_config)
        drops.extend(calls_drops)
        drops.extend(puts_drops)

    quotes: list[Quote] = []
    for _, row in calls_filtered.iterrows():
        qq = row_to_quote(row, OptionType.CALL)
        if qq is not None:
            quotes.append(qq)
    for _, row in puts_filtered.iterrows():
        qq = row_to_quote(row, OptionType.PUT)
        if qq is not None:
            quotes.append(qq)

    if not quotes:
        return None, [], drops

    sl_raw = ExpirySlice(expiry_time=T, quotes=quotes)
    kept, rejected = clean_quotes(sl_raw, spot)
    if not kept:
        return None, rejected, drops
    return ExpirySlice(expiry_time=T, quotes=kept), rejected, drops
