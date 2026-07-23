"""Frozen dataclass types for the mispricing backtest engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from arbfree_vol.models.option import OptionType


@dataclass(frozen=True, slots=True)
class MispricingSignal:
    """A single mispricing signal detected by comparing market IV to model IV.

    Parameters
    ----------
    strike:
        Absolute strike of the option.
    expiry_time:
        Time to expiry in years.
    expiry_date:
        Calendar expiry date (derived from snapshot_date + round(T * 365)).
    option_type:
        ``CALL`` or ``PUT``.
    market_iv:
        Implied volatility recovered from the market mid price.
    model_iv:
        Implied volatility from the fitted arbitrage-free surface.
    mispricing:
        ``market_iv - model_iv`` (signed; positive = overpriced relative to model).
    entry_price:
        Market mid price at the snapshot.
    side:
        Trade direction. ``+1`` = long the underpriced option,
        ``-1`` = short the overpriced option.
    """

    strike: float
    expiry_time: float
    expiry_date: date
    option_type: OptionType
    market_iv: float
    model_iv: float
    mispricing: float
    entry_price: float
    side: int


@dataclass(frozen=True, slots=True)
class Trade:
    """A delta-hedged trade entered on a mispricing signal.

    The frozen-vol convention means the hedge vol (= entry market_iv) is
    held constant for all daily rebalancings.
    """

    signal: MispricingSignal
    entry_date: date
    entry_spot: float
    frozen_vol: float
    quantity: int          # +1 long / -1 short (matches signal.side)
    risk_free: float
    div_yield: float


@dataclass(frozen=True, slots=True)
class TradePnL:
    """Realised P&L of a single delta-hedged trade held to expiry."""

    trade: Trade
    realized_pnl: float    # total delta-hedged P&L = option_pnl + hedge_pnl
    option_pnl: float      # qty * (intrinsic(expiry_spot) - entry_price)
    hedge_pnl: float       # cumulative delta-hedge P&L (signed by qty)
    expiry_spot: float
    hold_days: int
    hit: bool              # realized_pnl > 0


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Aggregated results for a single-cohort mispricing backtest."""

    trades: tuple[Trade, ...]
    pnls: tuple[TradePnL, ...]
    n_trades: int
    hit_rate: float
    total_pnl: float
    mean_pnl: float
    std_pnl: float
    sharpe: float           # mean / std per-trade; 0 if std == 0 or n < 2
    max_drawdown: float     # peak-to-trough on cumulative pnl (expiry-ordered)
    pnl_p5: float
    pnl_p50: float
    pnl_p95: float
