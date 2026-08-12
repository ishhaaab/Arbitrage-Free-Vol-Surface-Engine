"""Dupire local volatility heatmap visualization."""

import numpy as np
from matplotlib.figure import Figure

from arbfree_vol.pricing.local_vol import LocalVolSurface


def plot_dupire_heatmap(
    lv: LocalVolSurface,
    symbol: str = "SPY",
    fallback_slices: list[float] | None = None,
) -> Figure:
    """2-D heatmap of the Dupire local-volatility grid.

    Parameters
    ----------
    lv:
        ``LocalVolSurface`` frozen dataclass containing the local-vol
        grid (``.strikes``, ``.maturities``, ``.grid``).
    symbol:
        Ticker symbol for the plot title.
    fallback_slices:
        Optional list of T values that used the eSSVI fallback path.
        If provided, an annotation is added to the plot.  The actual
        row masking is driven by NaN values in the grid itself —
        ``dupire()`` propagates NaN into any row whose FD stencil
        touches a fallback slice.  This is the single source of truth
        for invalid cells.

    Returns
    -------
    Figure
    """
    import matplotlib

    strikes = np.array(lv.strikes)
    maturities = np.array(lv.maturities)
    grid = np.array(lv.grid)  # (n_maturities, n_strikes)

    # Mask based on NaN in the grid (set by dupire() for fallback-
    # contaminated rows and any other undefined cells).
    has_nan = np.any(np.isnan(grid))

    grid = np.ma.masked_invalid(grid)

    fig = Figure(figsize=(11, 7))
    ax = fig.add_subplot(111)

    cmap = matplotlib.colormaps["inferno"].copy()
    if has_nan:
        # with_extremes replaces the deprecated set_bad API (matplotlib >= 3.7
        # deprecation); the bad color is passed as an RGBA tuple because
        # with_extremes has no alpha keyword.
        cmap.with_extremes(bad=(0.5019607843137255, 0.5019607843137255,
                                0.5019607843137255, 0.5))

    mesh = ax.pcolormesh(strikes, maturities, grid,
                         cmap=cmap, shading="auto")

    cb = fig.colorbar(mesh, ax=ax, shrink=0.7, aspect=25, pad=0.02)
    cb.set_label("Local volatility")

    if fallback_slices:
        ax.text(
            0.02, 0.02,
            "Grayed region: non-monotonic ATM variance — see Issue #15",
            transform=ax.transAxes,
            fontsize=8,
            color="dimgray",
            verticalalignment="bottom",
        )

    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to expiry (yrs)")
    ax.set_title(f"{symbol} Dupire local volatility")

    fig.tight_layout()
    return fig
