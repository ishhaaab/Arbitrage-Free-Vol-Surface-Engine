"""3D surface plot of the fitted implied volatility surface."""

from arbfree_vol.svi.model import svi_total_variance
from arbfree_vol.repair.report import FittedSlice

from math import sqrt

import numpy as np
from matplotlib.figure import Figure



def plot_surface(
    fitted_slices: list[FittedSlice],
    n_k: int= 40,
) -> Figure:
    """Plot a 3D surface of fitted total variance over (T, k) space.

    Parameters:

        fitted_slices : list[FittedSlice]
            Sorted list of fitted slices which will be sorted by expiry_time.
        n_k : int
            Number of log moneyness points per slice.

    Returns:
    
        matplotlib.figure.Figure
    """
    ordered= sorted(fitted_slices, key=lambda fs: fs.expiry_time)
    if len(ordered) < 2:
        raise ValueError("Need at least two fitted slices for a surface plot")

    k_min= -1.5
    k_max= 1.5
    k_grid= np.linspace(k_min, k_max, n_k)
    T_vals= np.array([fs.expiry_time for fs in ordered])

    # build Z= total variance for each (T, k)
    Z= np.zeros((len(T_vals), n_k))
    for i, fs in enumerate(ordered):
        p= fs.params
        for j, k in enumerate(k_grid):
            Z[i, j]= svi_total_variance(k, p.a, p.b, p.rho, p.m, p.sigma)

    # convert total variance to implied vol for readability
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_surface= np.where(T_vals[:, None] > 0,
                                 np.sqrt(Z / T_vals[:, None]), 0.0)

    K_grid, T_mesh= np.meshgrid(k_grid, T_vals)

    fig= Figure(figsize=(10, 7))
    ax= fig.add_subplot(111, projection="3d")
    ax.plot_surface(T_mesh, K_grid, sigma_surface, cmap="viridis", alpha=0.85)

    ax.set_xlabel("Time to expiry (yrs)")
    ax.set_ylabel("Log-moneyness k")
    ax.set_zlabel("Implied volatility")
    ax.set_title("Fitted implied volatility surface")

    fig.tight_layout()
    return fig
