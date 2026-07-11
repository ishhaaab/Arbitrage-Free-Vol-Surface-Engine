"""3D surface plot of the fitted implied volatility surface."""

from arbfree_vol.svi.model import svi_total_variance
from arbfree_vol.repair.report import FittedSlice

from math import sqrt

import numpy as np
from matplotlib.figure import Figure
from matplotlib import cm


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation: ``a * (1 - t) + b * t``."""
    return a * (1.0 - t) + b * t


def _interpolate_params(
    T_target: float,
    earlier: FittedSlice,
    later: FittedSlice,
) -> tuple[float, float, float, float, float]:
    """Interpolate SVI params between two adjacent fitted slices at *T_target*.

    Clamps ``b >= 0`` and ``sigma > 0`` after interpolation.
    """
    t = (T_target - earlier.expiry_time) / (later.expiry_time - earlier.expiry_time)
    t = max(0.0, min(1.0, t))

    p_e = earlier.params
    p_l = later.params
    a = _lerp(p_e.a, p_l.a, t)
    b = max(_lerp(p_e.b, p_l.b, t), 1e-8)
    rho = _lerp(p_e.rho, p_l.rho, t)
    m = _lerp(p_e.m, p_l.m, t)
    sigma = max(_lerp(p_e.sigma, p_l.sigma, t), 1e-8)
    return a, b, rho, m, sigma


def plot_surface(
    fitted_slices: list[FittedSlice],
    n_k: int = 200,
    n_T: int = 150,
) -> Figure:
    """Make a 3D plot of the fitted SVI surface.

    Interpolates SVI parameters between slices so the surface is smooth
    even with only a handful fitted expiries.
    """
    ordered = sorted(fitted_slices, key=lambda fs: fs.expiry_time)
    if len(ordered) < 2:
        raise ValueError("Need at least two fitted slices for a surface plot")

    k_min, k_max = -1.5, 1.5
    k_grid = np.linspace(k_min, k_max, n_k)

    # Build a dense T grid that spans the fitted range
    T_start = ordered[0].expiry_time
    T_end = ordered[-1].expiry_time
    T_grid = np.linspace(T_start, T_end, n_T)

    Z = np.zeros((n_T, n_k))

    for i, T in enumerate(T_grid):
        if T <= ordered[0].expiry_time:
            a, b, rho, m, sigma = (ordered[0].params.a, ordered[0].params.b,
                                    ordered[0].params.rho, ordered[0].params.m,
                                    ordered[0].params.sigma)
        elif T >= ordered[-1].expiry_time:
            a, b, rho, m, sigma = (ordered[-1].params.a, ordered[-1].params.b,
                                    ordered[-1].params.rho, ordered[-1].params.m,
                                    ordered[-1].params.sigma)
        else:
            for j in range(len(ordered) - 1):
                if ordered[j].expiry_time <= T <= ordered[j + 1].expiry_time:
                    a, b, rho, m, sigma = _interpolate_params(
                        T, ordered[j], ordered[j + 1])
                    break

        for j, k in enumerate(k_grid):
            Z[i, j] = svi_total_variance(k, a, b, rho, m, sigma)

    # Convert total variance to implied vol for readability
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_surface = np.where(T_grid[:, None] > 0,
                                 np.sqrt(Z / T_grid[:, None]), 0.0)

    K_grid, T_mesh = np.meshgrid(k_grid, T_grid)

    fig = Figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Surface with subtle wireframe to show curvature, fully opaque
    surf = ax.plot_surface(T_mesh, K_grid, sigma_surface,
                            cmap="plasma", linewidth=0.15,
                            antialiased=True, alpha=1.0, edgecolor="black")

    # Colorbar to map colors to vol values
    cb = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cb.set_label("Implied volatility")

    ax.view_init(elev=28, azim=-55)
    ax.set_xlabel("Time to expiry (yrs)", labelpad=10)
    ax.set_ylabel("Log-moneyness k", labelpad=10)
    ax.set_zlabel("Implied volatility", labelpad=10)
    ax.set_title("Fitted implied volatility surface")

    fig.tight_layout()
    return fig
