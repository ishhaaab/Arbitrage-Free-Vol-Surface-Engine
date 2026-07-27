"""Portfolio Greeks heatmaps."""

import numpy as np
from matplotlib.figure import Figure

from arbfree_vol.models.option import OptionType
from arbfree_vol.surface.greeks import bucketed_greeks
from arbfree_vol.surface.interpolate import FittedSurface


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

