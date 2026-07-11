"""Fetch live option chains from yfinance and build a VolSurface.

Attempts to source real risk-free rates and dividend yields.  Falls
back to pre-pass forward-curve estimation when rates are unavailable.
"""

import math
from datetime import date
from typing import Any

import yfinance as yf

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.repair.fwd_curve import estimate_forward_curve
from arbfree_vol.ingestion.cleaning import clean_quotes, RejectionRecord


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
        pass
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
            # definitely in percent — divide by 100.
            if q > 0.50:
                q /= 100.0
            return q
    except Exception:
        pass
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
) -> tuple[VolSurface, list[RejectionRecord]]:
    """Fetch an option chain from yfinance and return a cleaned VolSurface.

    Steps through the nearest expiries, builds quotes from mid prices
    (not stale lastPrice), applies the cleaning layer (wide spreads,
    crossed markets, deep moneyness, near-expiry, zero prices), and
    returns the cleaned surface plus an audit trail of rejects.

    Real r comes from ``^IRX`` (13-week T-bill), real q from
    ``info.dividendYield``.  When either is unavailable, defaults to
    ``r=0.05, q=0.0`` — the repair pipeline's ``detect_with_forward()``
    corrects for that at detection time.
    """
    ticker = yf.Ticker(symbol)
    expiries = ticker.options

    if not expiries:
        raise ValueError(f"No expiries available for symbol {symbol!r}")

    # source rates
    r = _get_risk_free_rate()
    q = _get_dividend_yield(ticker)
    if r is None or q is None:
        # fallback — detect_with_forward() will correct via pre-pass
        r = r or 0.05
        q = q or 0.0

    # get the underlying spot price
    spot = None
    try:
        info = ticker.info or {}
        spot = info.get("regularMarketPrice") or info.get("previousClose")
    except Exception:
        pass
    if spot is None or not isinstance(spot, (int, float)):
        raise ValueError(f"Could not fetch spot price for {symbol!r}")

    spot = float(spot)

    # build slices from available expiries
    all_rejected: list[RejectionRecord] = []
    slices: list[ExpirySlice] = []
    ref_date = date.today()

    for exp_str in expiries:
        if len(slices) >= max_expiries:
            break

        T = (date.fromisoformat(exp_str) - ref_date).days / 365.0
        if T <= min_T_years:
            continue

        chain = ticker.option_chain(exp_str)
        quotes: list[Quote] = []

        for _, row in chain.calls.iterrows():
            qq = _row_to_quote(row, OptionType.CALL)
            if qq is not None:
                quotes.append(qq)

        for _, row in chain.puts.iterrows():
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

    return VolSurface(spot=spot, risk_free=r, div_yield=q, slices=slices), all_rejected
