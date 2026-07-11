from arbfree_vol.models.surface import VolSurface, get_r, get_q
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
    """Plot fitted SVI smiles with raw total-variance data beneath.

    One subplot per expiry: raw (k, w) as scatter, fitted SVI as a line.
    Helps you see how well each slice was fit.
    """
    by_T: dict[float, list[tuple[float, float]]]= {}
    for sl in surface.slices:
        strike_w= slice_total_variance(surface, sl)
        r= get_r(surface, sl)
        q= get_q(surface, sl)
        F= surface.spot * exp((r - q) * sl.expiry_time)
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


def plot_smiles_heatmap(
    fitted_slices: list[FittedSlice],
    n_k: int = 150,
    symbol: str = "SPY",
) -> Figure:
    """One-smile-per-row heatmap: T rows, k columns, color = implied vol.

    Each row corresponds to one fitted expiry.  The color shows the
    total variance (w) at that (k, T).  Missing cells are left empty.
    """
    from math import sqrt as _sqrt

    ordered = sorted(fitted_slices, key=lambda fs: fs.expiry_time)
    if len(ordered) < 1:
        raise ValueError("No fitted slices to plot")

    pts_by_T: dict[float, list[tuple[float, float]]] = {}
    for fs in ordered:
        if fs.data_points is None:
            continue
        pts_by_T[fs.expiry_time] = list(fs.data_points)

    if not pts_by_T:
        raise ValueError("No data points in any fitted slice")

    T_vals = sorted(pts_by_T.keys())

    k_min, k_max = -1.5, 1.5
    k_grid = np.linspace(k_min, k_max, n_k)

    # Build rows: each row is a (k_grid, w_interp) for one T
    rows: list[np.ndarray] = []
    for T in T_vals:
        pts = np.array(sorted(pts_by_T[T]))
        ks, ws = pts[:, 0], pts[:, 1]
        if len(ks) < 2:
            w_row = np.full(n_k, np.nan)
        else:
            w_row = np.interp(k_grid, ks, ws, left=np.nan, right=np.nan)
        rows.append(w_row)

    w_grid = np.ma.masked_invalid(np.array(rows))  # (n_T, n_k)

    fig = Figure(figsize=(11, 0.4 * len(T_vals) + 2))
    ax = fig.add_subplot(111)

    mesh = ax.pcolormesh(k_grid, T_vals, w_grid, cmap="plasma", shading="auto")

    cb = fig.colorbar(mesh, ax=ax, shrink=0.7, aspect=25, pad=0.02)
    cb.set_label("Total variance w")

    ax.set_xlabel("Log-moneyness k")
    ax.set_ylabel("Time to expiry T")
    ax.set_title(f"{symbol} per-slice smiles ({len(T_vals)} expiries)")

    fig.tight_layout()
    return fig
