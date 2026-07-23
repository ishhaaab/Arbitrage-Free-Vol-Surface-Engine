"""Backtest orchestration — run a complete mispricing backtest cohort."""

from __future__ import annotations

from datetime import date
from typing import Callable

import numpy as np

from arbfree_vol.models.surface import VolSurface
from arbfree_vol.surface.interpolate import FittedSurface

from arbfree_vol.backtest.prices import fetch_underlying_path
from arbfree_vol.backtest.signal import detect_mispricing, _MIN_SIGNAL_T_YEARS_DEFAULT
from arbfree_vol.backtest.pnl import realize_trade_pnl
from arbfree_vol.backtest.types import BacktestResult, Trade, TradePnL


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_backtest(
    surface: VolSurface,
    fs: FittedSurface,
    symbol: str,
    snapshot_date: date | None = None,
    threshold: float = 0.01,
    min_signal_T_years: float = _MIN_SIGNAL_T_YEARS_DEFAULT,
    fetch_prices: Callable[[str, date, date], dict[date, float]] | None = None,
) -> BacktestResult:
    """Run a single-cohort mispricing backtest.

    Algorithm
    ---------
    1. Detect mispricing signals via ``detect_mispricing``.
    2. Build ``Trade`` objects with the simpler per-trade r/q convention
       (all trades use the surface-level ``risk_free`` / ``div_yield``).
    3. Fetch the underlying price path from *snapshot_date* to the last
       expiry date among all signals.
    4. Realise P&L for each trade via ``realize_trade_pnl``.
    5. Aggregate metrics: hit rate, Sharpe, max drawdown, percentiles.

    Parameters
    ----------
    surface:
        Raw (or cleaned) volatility surface.
    fs:
        Fitted arbitrage-free surface.
    symbol:
        Ticker symbol for fetching underlying price data.
    snapshot_date:
        Trade entry date.  Defaults to ``date.today()``.
    threshold:
        Minimum mispricing (vol points) to trade.
    min_signal_T_years:
        Minimum time-to-expiry for a quote to generate a signal.
    fetch_prices:
        Callable ``(symbol, start, end) -> dict[date, float]``.
        Defaults to ``fetch_underlying_path`` (live yfinance data).
        Inject a stub or mock for testing.

    Returns
    -------
    BacktestResult
        Aggregated metrics.  If no signals are found, returns an empty
        result (all zeros, empty tuples).
    """
    snapshot_date = snapshot_date or date.today()

    signals = detect_mispricing(
        surface=surface,
        fs=fs,
        threshold=threshold,
        min_signal_T_years=min_signal_T_years,
        snapshot_date=snapshot_date,
    )

    if not signals:
        return _empty_result()

    # Build trades — use surface-level r/q (simpler approach per spec)
    trades: list[Trade] = [
        Trade(
            signal=sg,
            entry_date=snapshot_date,
            entry_spot=surface.spot,
            frozen_vol=sg.market_iv,
            quantity=sg.side,
            risk_free=surface.risk_free,
            div_yield=surface.div_yield,
        )
        for sg in signals
    ]

    last_expiry_date = max(sg.expiry_date for sg in signals)

    fetch_fn = (
        fetch_prices if fetch_prices is not None else fetch_underlying_path
    )
    price_path = fetch_fn(symbol, snapshot_date, last_expiry_date)

    pnls: list[TradePnL] = [
        realize_trade_pnl(tr, price_path) for tr in trades
    ]

    return _aggregate(trades, pnls)


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _empty_result() -> BacktestResult:
    return BacktestResult(
        trades=(),
        pnls=(),
        n_trades=0,
        hit_rate=0.0,
        total_pnl=0.0,
        mean_pnl=0.0,
        std_pnl=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
        pnl_p5=0.0,
        pnl_p50=0.0,
        pnl_p95=0.0,
    )


def _aggregate(
    trades: list[Trade], pnls: list[TradePnL]
) -> BacktestResult:
    n = len(pnls)
    realized = np.array([p.realized_pnl for p in pnls], dtype=float)

    # hit rate
    n_hits = sum(1 for p in pnls if p.hit)
    hit_rate = n_hits / n if n > 0 else 0.0

    # mean / std
    total_pnl = float(np.sum(realized))
    mean_pnl = total_pnl / n if n > 0 else 0.0
    std_pnl = float(np.std(realized, ddof=1)) if n > 1 else 0.0

    # Sharpe (per-trade)
    sharpe = mean_pnl / std_pnl if std_pnl > 0.0 and n >= 2 else 0.0

    # max drawdown — order by expiry_date
    sorted_pnls = sorted(pnls, key=lambda p: p.trade.signal.expiry_date)
    cumulative = np.cumsum(np.array([p.realized_pnl for p in sorted_pnls]))
    if len(cumulative) > 0:
        running_max = np.maximum.accumulate(cumulative)
        max_drawdown = float(np.max(running_max - cumulative))
    else:
        max_drawdown = 0.0

    # percentiles
    p5 = float(np.percentile(realized, 5)) if n >= 1 else 0.0
    p50 = float(np.percentile(realized, 50)) if n >= 1 else 0.0
    p95 = float(np.percentile(realized, 95)) if n >= 1 else 0.0

    return BacktestResult(
        trades=tuple(trades),
        pnls=tuple(pnls),
        n_trades=n,
        hit_rate=hit_rate,
        total_pnl=total_pnl,
        mean_pnl=mean_pnl,
        std_pnl=std_pnl,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        pnl_p5=p5,
        pnl_p50=p50,
        pnl_p95=p95,
    )
