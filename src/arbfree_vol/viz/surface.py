"""3D surface plot of the fitted implied volatility surface."""

from arbfree_vol.svi.model import svi_total_variance
from arbfree_vol.repair.report import FittedSlice

from math import sqrt

import numpy as np
from matplotlib.figure import Figure


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
    n_k: int = 100,
    n_T: int = 80,
) -> Figure:
    """Plot a smooth 3D surface of fitted total variance over (T, k) space.

    The T axis is upsampled via linear interpolation of SVI parameters
    between adjacent fitted slices so the surface appears smooth even
    when fitted maturities are sparse.

    Parameters
    ----------
    fitted_slices : list[FittedSlice]
        List of fitted slices (one per fitted expiry).
    n_k : int
        Number of log-moneyness grid points (k-axis).
    n_T : int
        Number of time grid points (T-axis) — interpolated between slices.

    Returns
    -------
    matplotlib.figure.Figure
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
            # Find the two adjacent slices bracketing T
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

    fig = Figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T_mesh, K_grid, sigma_surface, cmap="viridis", alpha=0.85)

    ax.set_xlabel("Time to expiry (yrs)")
    ax.set_ylabel("Log-moneyness k")
    ax.set_zlabel("Implied volatility")
    ax.set_title("Fitted implied volatility surface")

    fig.tight_layout()
    return fig
