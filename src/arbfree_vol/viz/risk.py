"""Portfolio Greeks heatmaps and scenario P&L visualizations."""

import numpy as np
from matplotlib.figure import Figure

from arbfree_vol.models.option import OptionType
from arbfree_vol.surface.greeks import bucketed_greeks
from arbfree_vol.surface.interpolate import FittedSurface
from arbfree_vol.surface.risk import ScenarioResult


def plot_greeks_heatmap(
    fs: FittedSurface,
    strikes: list[float],
    maturities: list[float],
    greek_names: tuple[str, ...] = ("delta", "gamma", "vega"),
    symbol: str = "SPY",
) -> Figure:
    """Heatmap grid of option Greeks over (strike, maturity) space.

    Parameters
    ----------
    fs:
        Fitted volatility surface.
    strikes:
        Strike grid (x-axis of each heatmap).
    maturities:
        Maturity grid (y-axis of each heatmap).
    greek_names:
        Which Greeks to display (subset of ``"delta"``, ``"gamma"``,
        ``"vega"``, ``"theta"``, ``"rho"``).
    symbol:
        Ticker symbol for the plot title.

    Returns
    -------
    Figure
    """
    greeks = bucketed_greeks(
        fs, strikes, maturities, OptionType.CALL,
        r=fs.risk_free, q=fs.div_yield,
    )

    n_greeks = len(greek_names)
    fig = Figure(figsize=(5 * n_greeks, 4))
    fig.suptitle(f"{symbol} Greeks (CALL)", fontsize=13)

    strike_mesh, T_mesh = np.meshgrid(strikes, maturities)

    for idx, name in enumerate(greek_names):
        ax = fig.add_subplot(1, n_greeks, idx + 1)
        data = np.ma.masked_invalid(greeks[name].T)
        mesh = ax.pcolormesh(strike_mesh, T_mesh, data,
                             cmap="RdYlBu_r", shading="auto")
        cb = fig.colorbar(mesh, ax=ax, shrink=0.7, aspect=25, pad=0.02)
        cb.set_label(name.capitalize())
        ax.set_xlabel("Strike")
        ax.set_ylabel("Time to expiry (yrs)")
        ax.set_title(name.capitalize())

    fig.tight_layout()
    return fig


def plot_scenario_payoff(
    scenarios: list[ScenarioResult],
    symbol: str = "SPY",
) -> Figure:
    """Bar chart of P&L under spot-bump scenarios with delta overlay.

    Parameters
    ----------
    scenarios:
        List of ``ScenarioResult`` from ``spot_bump_analysis``.
    symbol:
        Ticker symbol for the plot title.

    Returns
    -------
    Figure
    """
    bumps = [s.spot_bump for s in scenarios]
    pnls = [s.pnl for s in scenarios]
    delta_pnls = [s.delta_pnl for s in scenarios]

    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Colour bars by sign
    bar_colors = ["crimson" if p < 0 else "seagreen" for p in pnls]
    labels = [f"{b:+.0%}" for b in bumps]
    bars = ax.bar(labels, pnls, color=bar_colors, alpha=0.8, label="P&L")

    # Delta approximation overlay
    ax.plot(labels, delta_pnls, "b--", linewidth=1.5, marker="o",
            label="Delta approx")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Spot bump")
    ax.set_ylabel("P&L")
    ax.set_title(f"{symbol} spot-bump scenario P&L")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
