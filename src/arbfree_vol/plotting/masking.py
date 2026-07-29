"""Masking utilities for fallback-derived cells in vol surface plots.

When the eSSVI sequential hard-constrained fit fails for a maturity slice
(due to non-monotonic ATM total variance), the engine falls back to the
unconstrained per-slice fit.  These ``fallback_slices`` are NOT
arbitrage-free by construction, and visualizations should clearly mark
them so consumers don't mistake model artifacts for calibrated values.
"""

import numpy as np
from numpy.typing import NDArray


def make_fallback_mask(
    grid_T: NDArray[np.floating],
    fallback_slices: list[float],
    tol: float = 0.01,
) -> NDArray[np.bool_]:
    """Return a boolean mask marking grid columns that fall on fallback slices.

    Parameters
    ----------
    grid_T:
        1-D array of T (time-to-expiry) values used as the maturity axis
        in a plot grid.  May come from a linspace, from the surface
        ``maturities`` list, or from a heat-map's y-axis.
    fallback_slices:
        The list of T values reported by ``RepairReport.fallback_slices``
        — maturity times where the eSSVI hard-constrained fit failed and
        only the unconstrained per-slice fit succeeded.
    tol:
        Maximum absolute distance between a grid T value and a fallback
        T value for the grid value to be considered a match.

    Returns
    -------
    NDArray[np.bool_]
        Boolean array of the same length as ``grid_T``.
        ``True`` at index *i* means ``grid_T[i]`` is within ``tol`` of
        at least one entry in ``fallback_slices``.
    """
    grid_T = np.asarray(grid_T, dtype=np.float64)
    mask = np.zeros(grid_T.shape, dtype=bool)

    if not fallback_slices:
        return mask

    fb = np.asarray(fallback_slices, dtype=np.float64)

    # For each grid point, check if any fallback T is close enough.
    # Broadcasting: grid_T[:, None] vs fb[None, :] -> (n_grid, n_fb)
    distances = np.abs(grid_T[:, None] - fb[None, :])
    mask = np.any(distances <= tol, axis=1)

    return mask
