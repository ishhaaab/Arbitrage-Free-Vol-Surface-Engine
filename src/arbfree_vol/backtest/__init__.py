"""Mispricing backtest engine.

Provides a single-cohort cross-sectional mispricing backtest:

1. Detect quotes whose market IV differs from the fitted arbitrage-free surface.
2. Enter delta-hedged trades (long underpriced, short overpriced).
3. Realise P&L at expiry under the frozen-vol hedge convention.
4. Aggregate metrics: hit rate, Sharpe, max drawdown, percentiles.
"""

from arbfree_vol.backtest.types import (
    BacktestResult,
    MispricingSignal,
    Trade,
    TradePnL,
)
from arbfree_vol.backtest.signal import detect_mispricing
from arbfree_vol.backtest.pnl import realize_trade_pnl
from arbfree_vol.backtest.engine import run_backtest
from arbfree_vol.backtest.prices import fetch_underlying_path

__all__ = [
    "MispricingSignal",
    "Trade",
    "TradePnL",
    "BacktestResult",
    "detect_mispricing",
    "realize_trade_pnl",
    "run_backtest",
    "fetch_underlying_path",
]
