"""Backtest visualisation — P&L distribution, cumulative P&L, mispricing scatter, metrics summary.

All functions return a ``matplotlib.figure.Figure`` and handle empty results
(``n_trades == 0``) by rendering a "No trades" message on a single axes.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from arbfree_vol.backtest.types import BacktestResult


def _empty_figure(message: str = "No trades") -> Figure:
    """Return a single-axes figure with a centred *message*."""
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.tight_layout()
    return fig


def plot_pnl_distribution(result: BacktestResult, symbol: str = "SPY") -> Figure:
    """Histogram of per-trade realised P&L with vertical lines at 0 and the mean.

    Parameters
    ----------
    result:
        Aggregated backtest result.
    symbol:
        Ticker symbol for the plot title.

    Returns
    -------
    Figure
    """
    if result.n_trades == 0:
        return _empty_figure()

    pnls = np.array([p.realized_pnl for p in result.pnls], dtype=float)

    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(pnls, bins="auto", color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0, label="Zero P&L")
    ax.axvline(
        result.mean_pnl,
        color="crimson",
        linestyle="-",
        linewidth=1.5,
        label=f"Mean = {result.mean_pnl:.3f}",
    )
    ax.set_xlabel("Realised P&L ($)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{symbol} per-trade P&L distribution")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_cumulative_pnl(result: BacktestResult, symbol: str = "SPY") -> Figure:
    """Cumulative P&L curve ordered by trade expiry date.

    The max-drawdown region is highlighted when ``result.max_drawdown > 0``.

    Parameters
    ----------
    result:
        Aggregated backtest result.
    symbol:
        Ticker symbol for the plot title.

    Returns
    -------
    Figure
    """
    if result.n_trades == 0:
        return _empty_figure()

    # Sort by expiry date
    sorted_pairs = sorted(
        result.pnls,
        key=lambda p: p.trade.signal.expiry_date,
    )
    dates = [p.trade.signal.expiry_date for p in sorted_pairs]
    realized = np.array([p.realized_pnl for p in sorted_pairs], dtype=float)
    cumulative = np.cumsum(realized)

    fig = Figure(figsize=(9, 5))
    ax = fig.add_subplot(111)
    ax.plot(dates, cumulative, color="steelblue", marker=".", linewidth=1.5)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Expiry date")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_title(f"{symbol} cumulative P&L")

    # Highlight max-drawdown region
    if result.max_drawdown > 0.0 and len(cumulative) > 1:
        running_max = np.maximum.accumulate(cumulative)
        dd = running_max - cumulative
        dd_peak_idx = int(np.argmax(dd))
        if dd[dd_peak_idx] > 0.0:
            trough_val = cumulative[dd_peak_idx]
            peak_val = running_max[dd_peak_idx]
            ax.fill_between(
                dates,
                cumulative,
                running_max,
                where=(cumulative < running_max),
                color="crimson",
                alpha=0.12,
                label=f"Max DD = {result.max_drawdown:.2f}",
            )
            ax.axvline(dates[dd_peak_idx], color="crimson", linestyle=":", linewidth=0.8)
            ax.legend(fontsize=9)

    fig.tight_layout()
    return fig


def plot_mispricing_vs_pnl(result: BacktestResult, symbol: str = "SPY") -> Figure:
    """Scatter of mispricing (vol points) vs realised P&L, coloured by trade side.

    Parameters
    ----------
    result:
        Aggregated backtest result.
    symbol:
        Ticker symbol for the plot title.

    Returns
    -------
    Figure
    """
    if result.n_trades == 0:
        return _empty_figure()

    mispricing = np.array(
        [p.trade.signal.mispricing for p in result.pnls], dtype=float
    )
    realized = np.array([p.realized_pnl for p in result.pnls], dtype=float)
    sides = np.array([p.trade.signal.side for p in result.pnls], dtype=int)

    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    long_mask = sides == 1
    short_mask = sides == -1

    if np.any(long_mask):
        ax.scatter(
            mispricing[long_mask],
            realized[long_mask],
            color="seagreen",
            alpha=0.7,
            edgecolors="none",
            label="Long (underpriced)",
        )
    if np.any(short_mask):
        ax.scatter(
            mispricing[short_mask],
            realized[short_mask],
            color="crimson",
            alpha=0.7,
            edgecolors="none",
            label="Short (overpriced)",
        )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mispricing (market IV − model IV, vol pts)")
    ax.set_ylabel("Realised P&L ($)")
    ax.set_title(f"{symbol} mispricing vs realised P&L")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_backtest_metrics(result: BacktestResult, symbol: str = "SPY") -> Figure:
    """Summary panel of key backtest metrics as a 2×2 grid.

    Panels
    ------
    1. Number of trades and hit rate.
    2. Total P&L.
    3. Sharpe ratio.
    4. P&L percentiles (P5, P50, P95).

    Parameters
    ----------
    result:
        Aggregated backtest result.
    symbol:
        Ticker symbol for the plot title.

    Returns
    -------
    Figure
    """
    if result.n_trades == 0:
        return _empty_figure()

    fig = Figure(figsize=(10, 6))
    fig.suptitle(f"{symbol} backtest metrics", fontsize=14)

    # Panel 1 — n_trades & hit_rate
    ax1 = fig.add_subplot(221)
    labels = ["Trades", "Hit rate"]
    values = [result.n_trades, result.hit_rate * 100.0]
    bars1 = ax1.bar(labels, values, color=["steelblue", "seagreen"], width=0.5)
    ax1.set_ylabel("Count / %")
    ax1.set_title("Activity")
    for bar, val in zip(bars1, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.1f}" if isinstance(val, float) else str(val),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Panel 2 — total_pnl
    ax2 = fig.add_subplot(222)
    color2 = "seagreen" if result.total_pnl >= 0.0 else "crimson"
    ax2.bar(["Total P&L"], [result.total_pnl], color=color2, width=0.4)
    ax2.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("P&L ($)")
    ax2.set_title(f"Total = ${result.total_pnl:.2f}")
    ax2.text(
        0,
        result.total_pnl,
        f"${result.total_pnl:.2f}",
        ha="center",
        va="bottom" if result.total_pnl >= 0.0 else "top",
        fontsize=10,
    )

    # Panel 3 — sharpe
    ax3 = fig.add_subplot(223)
    color3 = "seagreen" if result.sharpe >= 0.0 else "crimson"
    ax3.bar(["Sharpe"], [result.sharpe], color=color3, width=0.4)
    ax3.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_ylabel("Ratio")
    ax3.set_title(f"Sharpe = {result.sharpe:.2f}")

    # Panel 4 — P&L percentiles
    ax4 = fig.add_subplot(224)
    p_labels = ["P5", "P50", "P95"]
    p_values = [result.pnl_p5, result.pnl_p50, result.pnl_p95]
    colors4 = ["crimson", "steelblue", "seagreen"]
    bars4 = ax4.bar(p_labels, p_values, color=colors4, width=0.5)
    ax4.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax4.set_ylabel("P&L ($)")
    ax4.set_title("P&L percentiles")
    for bar, val in zip(bars4, p_values):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${val:.2f}",
            ha="center",
            va="bottom" if val >= 0.0 else "top",
            fontsize=9,
        )

    fig.tight_layout()
    return fig
