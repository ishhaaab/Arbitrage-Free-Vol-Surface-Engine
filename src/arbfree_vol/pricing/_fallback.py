"""Fallback-slice contamination guards for Dupire local volatility.

A fallback slice is one whose eSSVI sequential fit used the unconstrained
fallback path (its total variance is not trustworthy).  The Dupire
finite-difference stencil for ``dw/dT`` must not evaluate within an
interpolation interval that borders a fallback slice, or the computed
local vol leaks contaminated derivative values into neighbouring grid
rows.  This module owns that guard so the grid builder in
``local_vol.py`` stays a thin loop.

``_FB_TOL`` lives here because both the stencil guard and the grid
builder's sub-2-slice masking check need the same matching tolerance.
"""

import bisect

# tolerance for matching fallback T values
_FB_TOL: float = 0.01


def _stencil_touches_fallback(
    T: float,
    fitted_times: tuple[float, ...],
    fallback_set: set[float],
    dT: float,
) -> bool:
    """Check if T's Dupire FD stencil crosses or touches a fallback slice.

    The finite-difference stencil for dw/dT evaluates total variance at
    T-dT and T+dT.  If either of those points falls in an interpolation
    interval whose endpoint is a fallback slice, the computed derivative
    is contaminated.

    Parameters
    ----------
    T:
        Grid maturity to check.
    fitted_times:
        Sorted tuple of all fitted slice expiry times.
    fallback_set:
        Set of fallback T values, scanned linearly for proximity to the
        grid maturity and the stencil endpoints.
    dT:
        Finite-difference step used by ``_dw_dT``.

    Returns
    -------
    bool
        ``True`` if the stencil is contaminated by a fallback slice.
    """
    # T itself is a fallback slice
    for fb in fallback_set:
        if abs(T - fb) < _FB_TOL:
            return True

    # No interior interval exists to contaminate: with fewer than two
    # fitted slices there is no fitted_times[i+1] to bracket a stencil
    # point, so a fallback slice cannot leak through this path.  (The
    # fallback-maturity NaN row logic above is unchanged.)
    if len(fitted_times) < 2:
        return False

    # Check each FD stencil point: T - dT and T + dT
    n = len(fitted_times)
    for T_eval in (T - dT, T + dT):
        # Binary search: find index i such that
        # fitted_times[i] <= T_eval <= fitted_times[i+1]
        idx = bisect.bisect_right(fitted_times, T_eval) - 1
        idx = max(0, min(idx, n - 2))
        lo = fitted_times[idx]
        hi = fitted_times[idx + 1]
        for fb in fallback_set:
            if abs(lo - fb) < _FB_TOL or abs(hi - fb) < _FB_TOL:
                return True

    return False
