"""3D viewer (per-slice model ribbons + data scatter) and 2D heatmap."""

from math import sqrt

import numpy as np
from matplotlib.figure import Figure
from matplotlib import cm
from scipy.interpolate import griddata

from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.svi.model import svi_total_variance


def plot_surface(
    fitted_slices: list[FittedSlice],
    n_k: int = 200,
) -> Figure:
    """3D view of per-expiry fitted SVI curves with raw data on top.

    Each fitted slice is plotted as its own smooth SVI curve (ribbon).
    Actual (k, vol) data points are overlaid as scatter.  Nothing is
    interpolated between slices — what you see is what was fit.
    """
    ordered = sorted(fitted_slices, key=lambda fs: fs.expiry_time)
    if len(ordered) < 2:
        raise ValueError("Need at least two fitted slices to render")

    k_grid = np.linspace(-1.5, 1.5, n_k)
    T_min = ordered[0].expiry_time
    T_max = ordered[-1].expiry_time
    cmap = cm.viridis

    fig = Figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    for fs in ordered:
        p = fs.params
        ws = [svi_total_variance(float(k), p.a, p.b, p.rho, p.m, p.sigma)
              for k in k_grid]
        with np.errstate(divide="ignore", invalid="ignore"):
            vols = [sqrt(w / fs.expiry_time) if w > 0 else 0.0 for w in ws]

        T_arr = [fs.expiry_time] * n_k
        t_norm = (fs.expiry_time - T_min) / (T_max - T_min) if T_max > T_min else 0.0
        ax.plot(k_grid, T_arr, vols, color=cmap(t_norm), alpha=0.6, linewidth=1)

        if fs.data_points:
            ks_data = [float(k) for k, w in fs.data_points]
            ws_data = [float(w) for k, w in fs.data_points]
            vols_data = [sqrt(w / fs.expiry_time) if w > 0 else 0.0 for w in ws_data]
            ax.scatter(ks_data, [fs.expiry_time] * len(ks_data), vols_data,
                       color="crimson", edgecolors="black", linewidths=0.3,
                       s=20, alpha=0.9, zorder=5)

    ax.set_xlabel("Log-moneyness k", labelpad=10)
    ax.set_ylabel("Time to expiry (yrs)", labelpad=10)
    ax.set_zlabel("Implied volatility", labelpad=10)
    ax.set_title(f"Fitted SVI per-expiry smiles ({len(ordered)} expiries)")

    fig.tight_layout()
    return fig


def _build_point_cloud(fitted_slices: list[FittedSlice],
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Build a (k, T, vol) point cloud from fitted slice data points.

    Returns None if fewer than 5 points exist across all slices.
    """
    pts: list[tuple[float, float, float]] = []
    for fs in fitted_slices:
        if fs.data_points is None:
            continue
        for k, w in fs.data_points:
            if w > 0 and fs.expiry_time > 0:
                vol = sqrt(w / fs.expiry_time)
                pts.append((float(k), float(fs.expiry_time), float(vol)))

    if len(pts) < 5:
        return None

    pts_arr = np.array(pts)
    return pts_arr[:, 0], pts_arr[:, 1], pts_arr[:, 2]


def plot_heatmap_2d(
    fitted_slices: list[FittedSlice],
    n_k: int = 200,
    n_T: int = 150,
    symbol: str = "SPY",
    source: str = "yfinance",
) -> Figure:
    """2D heatmap of implied vol over (T, k) — the industry-standard view.

    Each cell shows the fitted implied vol.  Cells outside the data
    convex hull are left empty.
    """
    ordered = sorted(fitted_slices, key=lambda fs: fs.expiry_time)
    if len(ordered) < 2:
        raise ValueError("Need at least two fitted slices for a heatmap")

    cloud = _build_point_cloud(ordered)
    if cloud is None:
        raise ValueError("Not enough data points to render the heatmap")

    k_vals, T_vals, vol_vals = cloud

    k_min, k_max = -1.5, 1.5
    T_start = ordered[0].expiry_time
    T_end = ordered[-1].expiry_time

    k_grid = np.linspace(k_min, k_max, n_k)
    T_grid = np.linspace(T_start, T_end, n_T)
    K_grid, T_mesh = np.meshgrid(k_grid, T_grid)

    vol_grid = griddata((k_vals, T_vals), vol_vals,
                        (K_grid, T_mesh), method="cubic")
    vol_grid = np.ma.masked_invalid(vol_grid)

    fig = Figure(figsize=(11, 7))
    ax = fig.add_subplot(111)

    mesh = ax.pcolormesh(T_grid, k_grid, vol_grid.T,
                          cmap="plasma", shading="auto")

    cb = fig.colorbar(mesh, ax=ax, shrink=0.7, aspect=25, pad=0.02)
    cb.set_label("Implied volatility")

    ax.scatter(T_vals, k_vals, c="white", edgecolors="black",
               s=10, alpha=0.5, zorder=3)

    ax.set_xlabel("Time to expiry (yrs)")
    ax.set_ylabel("Log-moneyness k")
    ax.set_title(f"{symbol} implied volatility surface "
                 f"({len(ordered)} expiries, {source})")

    fig.tight_layout()
    return fig
