"""Dupire local volatility heatmap visualization."""

import numpy as np
from matplotlib.figure import Figure

from arbfree_vol.pricing.local_vol import LocalVolSurface


def plot_dupire_heatmap(
    lv: LocalVolSurface,
    symbol: str = "SPY",
) -> Figure:
    """2-D heatmap of the Dupire local-volatility grid.

    Parameters
    ----------
    lv:
        ``LocalVolSurface`` frozen dataclass containing the local-vol
        grid (``.strikes``, ``.maturities``, ``.grid``).
    symbol:
        Ticker symbol for the plot title.

    Returns
    -------
    Figure
    """
    strikes = np.array(lv.strikes)
    maturities = np.array(lv.maturities)
    grid = np.array(lv.grid)  # (n_maturities, n_strikes)
    grid = np.ma.masked_invalid(grid)

    fig = Figure(figsize=(11, 7))
    ax = fig.add_subplot(111)

    mesh = ax.pcolormesh(strikes, maturities, grid,
                         cmap="inferno", shading="auto")

    cb = fig.colorbar(mesh, ax=ax, shrink=0.7, aspect=25, pad=0.02)
    cb.set_label("Local volatility")

    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to expiry (yrs)")
    ax.set_title(f"{symbol} Dupire local volatility")

    fig.tight_layout()
    return fig
