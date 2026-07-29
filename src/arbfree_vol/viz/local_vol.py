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
        If provided, those maturity rows are grayed out in the plot
        and an annotation is added.

    Returns
    -------
    Figure
    """
    from arbfree_vol.plotting.masking import make_fallback_mask
    import matplotlib

    strikes = np.array(lv.strikes)
    maturities = np.array(lv.maturities)
    grid = np.array(lv.grid)  # (n_maturities, n_strikes)

    # Apply fallback mask: mark entire maturity rows as NaN
    if fallback_slices:
        fb_mask_1d = make_fallback_mask(maturities, fallback_slices)
        fb_mask_2d = fb_mask_1d[:, None] & np.ones(len(strikes), dtype=bool)
        grid = np.where(fb_mask_2d, np.nan, grid)

    grid = np.ma.masked_invalid(grid)

    fig = Figure(figsize=(11, 7))
    ax = fig.add_subplot(111)

    cmap = matplotlib.colormaps["inferno"].copy()
    if fallback_slices:
        cmap.set_bad("gray", alpha=0.5)

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
