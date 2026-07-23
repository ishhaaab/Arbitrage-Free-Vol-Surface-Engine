"""Underlying price path fetcher for the mispricing backtest.

.. note::

    ``yfinance`` is imported **lazily** inside ``fetch_underlying_path`` so the
    backtest package can be imported without a hard dependency on yfinance.
    This module is the mockable seam for tests — inject a custom ``fetch_prices``
    callable into ``run_backtest`` to bypass network calls.
"""

from __future__ import annotations

from datetime import date, timedelta


def fetch_underlying_path(
    symbol: str, start: date, end: date
) -> dict[date, float]:
    """Fetch daily closing prices for *symbol* from yfinance over [start, end].

    Parameters
    ----------
    symbol:
        Ticker symbol (e.g. ``"SPY"``).
    start:
        First date (inclusive).
    end:
        Last date (inclusive).  yfinance's ``history`` uses an exclusive end,
        so we pass ``end + timedelta(days=1)`` internally.

    Returns
    -------
    dict[date, float]
        Map from trading date to closing price.

    Raises
    ------
    ValueError
        If no data is returned by yfinance.
    """
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    # yfinance end is exclusive; add a day to include the final date
    df = ticker.history(start=start, end=end + timedelta(days=1))

    if df.empty:
        raise ValueError(
            f"No price data for {symbol!r} from {start} to {end}"
        )

    result: dict[date, float] = {}
    for idx, row in df.iterrows():
        result[idx.date()] = float(row["Close"])
    return result
