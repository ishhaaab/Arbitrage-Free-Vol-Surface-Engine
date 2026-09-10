"""Domain-limited numerical checks for fitted SVI surfaces."""

import numpy as np

from arbfree_vol.models.fitted import FittedSlice
from arbfree_vol.report import NumericalCertificate
from arbfree_vol.svi.model import svi_g, svi_total_variance


def verify_surface(
    slices: list[FittedSlice] | tuple[FittedSlice, ...],
    *,
    k_min: float = -1.5,
    k_max: float = 1.5,
    grid_size: int = 241,
    tolerance: float = 1e-4,
    has_fallbacks: bool = False,
    has_failures: bool = False,
) -> NumericalCertificate:
    """Check variance, butterfly density, and calendar margins on one grid.

    The result is a discrete certificate on ``[k_min, k_max]``. It is not
    an analytic or global no-arbitrage proof.
    """
    if k_min >= k_max:
        raise ValueError("k_min must be less than k_max")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    ordered = sorted(slices, key=lambda item: item.expiry_time)
    grid = np.linspace(k_min, k_max, grid_size)
    min_variance = float("inf")
    min_density = float("inf")
    min_calendar: float | None = None
    previous: np.ndarray | None = None

    for fitted in ordered:
        p = fitted.params
        variances = np.array(
            [svi_total_variance(float(k), p.a, p.b, p.rho, p.m, p.sigma) for k in grid]
        )
        densities = np.array(
            [svi_g(float(k), p.a, p.b, p.rho, p.m, p.sigma) for k in grid]
        )
        min_variance = min(min_variance, float(np.min(variances)))
        min_density = min(min_density, float(np.min(densities)))
        if previous is not None:
            pair_margin = float(np.min(variances - previous))
            min_calendar = pair_margin if min_calendar is None else min(min_calendar, pair_margin)
        previous = variances

    finite = bool(
        ordered
        and np.isfinite(min_variance)
        and np.isfinite(min_density)
        and (min_calendar is None or np.isfinite(min_calendar))
    )
    certified = bool(
        finite
        and min_variance >= -tolerance
        and min_density >= -tolerance
        and (min_calendar is None or min_calendar >= -tolerance)
        and not has_fallbacks
        and not has_failures
    )
    return NumericalCertificate(
        k_min=k_min,
        k_max=k_max,
        grid_size=grid_size,
        tolerance=tolerance,
        min_total_variance=min_variance,
        min_butterfly_density=min_density,
        min_calendar_margin=min_calendar,
        certified=certified,
    )
