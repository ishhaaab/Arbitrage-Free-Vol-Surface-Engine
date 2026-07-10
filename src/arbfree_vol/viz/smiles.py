from arbfree_vol.models.surface import VolSurface
from arbfree_vol.svi.model import svi_total_variance
from arbfree_vol.variance import slice_total_variance
from arbfree_vol.repair.report import FittedSlice

from math import log, exp

import numpy as np
from matplotlib.figure import Figure




def plot_smiles(
    surface: VolSurface,
    fitted_slices: list[FittedSlice],
    n_k: int= 80,
) -> Figure:
    """Plot fitted SVI smiles overlaid with observed total-variance points.

    One subplot per expiry slice.  Each subplot shows:
      - scatter points (k, w) from the raw surface
      - the fitted SVI curve as a continuous line

    Returns
    -------
    matplotlib.figure.Figure
    """
    by_T: dict[float, list[tuple[float, float]]]= {}
    for sl in surface.slices:
        strike_w= slice_total_variance(surface, sl)
        F= surface.spot * exp((surface.risk_free - surface.div_yield) * sl.expiry_time)
        pts= [(log(K / F), w) for K, w in strike_w.items()]
        by_T[sl.expiry_time]= sorted(pts)

    fitted_by_T= {fs.expiry_time: fs for fs in fitted_slices}
    T_vals= sorted(set(list(by_T.keys()) + list(fitted_by_T.keys())))

    if not T_vals:
        raise ValueError("No data to plot")

    n= len(T_vals)
    fig= Figure(figsize=(12, 4 * n))
    k_wide= np.linspace(-1.5, 1.5, n_k)

    for idx, T in enumerate(T_vals):
        ax= fig.add_subplot(n, 1, idx + 1)

        # raw data points
        if T in by_T:
            ks_raw, ws_raw= zip(*by_T[T])
            ax.scatter(ks_raw, ws_raw, color="steelblue", s=20, label="Observed w")

        # fitted curve
        if T in fitted_by_T:
            fs= fitted_by_T[T]
            p= fs.params
            ws_fit= [svi_total_variance(k, p.a, p.b, p.rho, p.m, p.sigma) for k in k_wide]
            ax.plot(k_wide, ws_fit, color="crimson", linewidth=1.5, label="SVI fit")

        ax.set_xlabel("Log-moneyness k")
        ax.set_ylabel("Total variance w")
        ax.set_title(f"T= {T:.2f} yr")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
