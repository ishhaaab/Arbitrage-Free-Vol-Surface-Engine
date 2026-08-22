"""Portfolio Greeks heatmaps."""

import numpy as np
from matplotlib.figure import Figure

from arbfree_vol.models.option import OptionType
from arbfree_vol.plotting.masking import (
    FALLBACK_BAD_RGBA,
    fallback_legend_handle,
)
from arbfree_vol.surface.greeks import bucketed_greeks
from arbfree_vol.surface.interpolate import FittedSurface


def plot_greeks_heatmap(
    fs: FittedSurface,
    strikes: list[float],
    maturities: list[float],
    greek_names: tuple[str, ...] = ("delta", "gamma", "vega"),
    symbol: str = "SPY",
    fallback_slices: list[float] | None = None,
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
    fallback_slices:
        Optional list of T values that used the eSSVI fallback path.
        If provided, those maturity rows are grayed out in the heatmap
        If provided, a legend explaining the gray rows is added under the
        plot heading.

    Returns
    -------
    Figure
    """
    from arbfree_vol.plotting.masking import make_fallback_mask
    import matplotlib

    greeks = bucketed_greeks(
        fs, strikes, maturities, OptionType.CALL,
        r=fs.risk_free, q=fs.div_yield,
    )

    n_greeks = len(greek_names)
    fig = Figure(figsize=(5 * n_greeks, 4))
    fig.suptitle(f"{symbol} Greeks (CALL)", fontsize=13)

    strike_mesh, T_mesh = np.meshgrid(strikes, maturities)

    # Pre-compute fallback mask (1-D over maturities)
    fb_mask_1d = None
    if fallback_slices:
        fb_mask_1d = make_fallback_mask(
            np.asarray(maturities), fallback_slices
        )

    for idx, name in enumerate(greek_names):
        ax = fig.add_subplot(1, n_greeks, idx + 1)
        data = greeks[name].T  # shape: (n_maturities, n_strikes)

        # Apply fallback mask: set entire maturity rows to NaN
        if fb_mask_1d is not None:
            fb_mask_2d = fb_mask_1d[:, None] & np.ones(len(strikes), dtype=bool)
            data = np.where(fb_mask_2d, np.nan, data)

        data = np.ma.masked_invalid(data)

        cmap = matplotlib.colormaps["RdYlBu_r"].copy()
        if fallback_slices:
            # with_extremes replaces the deprecated set_bad API (matplotlib
            # >= 3.7 deprecation). It returns a new colormap rather than
            # mutating in place, so the result must be assigned back.  The
            # bad RGBA is the shared fallback gray -- same constant the
            # legend swatch uses.
            cmap = cmap.with_extremes(bad=FALLBACK_BAD_RGBA)

        mesh = ax.pcolormesh(strike_mesh, T_mesh, data,
                             cmap=cmap, shading="auto")
        cb = fig.colorbar(mesh, ax=ax, shrink=0.7, aspect=25, pad=0.02)
        cb.set_label(name.capitalize())
        ax.set_xlabel("Strike")
        ax.set_ylabel("Time to expiry (yrs)")
        ax.set_title(name.capitalize())

    # ONE figure-level fallback legend for all panels, placed in the band
    # between the suptitle and the panel titles (right-aligned so it never
    # collides with the centered suptitle).  tight_layout is given a rect
    # that reserves this band when a legend exists.
    if fallback_slices:
        fig.legend(
            handles=[fallback_legend_handle()],
            loc="upper right",
            bbox_to_anchor=(0.995, 0.985),
            frameon=False,
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.90))
    else:
        fig.tight_layout()
    return fig

