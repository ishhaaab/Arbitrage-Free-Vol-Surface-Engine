"""Dupire local volatility from a fitted (SVI/SSVI) vol surface.

Provides
--------
- :func:`dupire_at` — local volatility at a single (K, T) point.
- :func:`dupire` —  full grid as a :class:`LocalVolSurface` frozen dataclass.
- :class:`LocalVolSurface` — tuple-based frozen container for the grid.

Implementation follows Gatheral (2004) eq. 1.10, computing the Dupire
local variance from total variance :math:`w(K,T) = \\sigma_\\text{imp}^2 T`:

.. math::

    \\sigma_\\text{loc}^2(K,T) =
    \\frac{ \\partial_T w }
         { 1 - \\frac{k}{w}\\partial_k w
           + \\frac14\\bigl(-\\frac1{\\sqrt{w}} - 1 + \\frac{k^2}{w}\\bigr)
             (\\partial_k w)^2
           + \\frac12 \\partial_{kk} w }

where :math:`k = \\ln(K / F(T))` and all partial derivatives are
approximated with finite differences.

.. note::
   Finite-difference step sizes are module-level constants
   (``_FD_T_DEFAULT``, ``_FD_K_DEFAULT``).  Because
   ``total_variance_at`` is piecewise-linear in *T* with kinks at
   slice expiries, the central FD in *T* can straddle a kink when the
   query maturity sits within ``dT`` of a slice expiry — producing
   minor FD jitter at slice boundaries.  Callers requiring
   boundary-clean local vols should query interior maturities or adapt
   ``dT`` to avoid straddling the nearest slice expiry.
"""

from dataclasses import dataclass
from math import log, nan, sqrt, isnan

import numpy as np

from arbfree_vol.surface.interpolate import FittedSurface, total_variance_at

# ---------------------------------------------------------------------------
# Module-level tolerances (no-hardcoding rule)
# ---------------------------------------------------------------------------
_FD_T_DEFAULT: float = 1e-3       # default finite-difference step in T (years)
_FD_K_DEFAULT: float = 1e-3       # default absolute strike step (used when
                                  # caller does not pass dK explicitly)
_DENOM_MIN: float = 1e-10         # denominator values ≤ this → local-vol
                                  # undefined (return nan)
_CAL_ARB_TOL: float = 0.0         # dw/dT ≤ tol → calendar arbitrage → raise
_T_MIN: float = 1e-4              # absolute tiny threshold for T near zero


# ---------------------------------------------------------------------------
# LocalVolSurface — compute boundary type (frozen dataclass, no Pydantic)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LocalVolSurface:
    """Frozen container for a Dupire local-volatility grid.

    Attributes
    ----------
    strikes:
        Sorted tuple of absolute strikes.
    maturities:
        Sorted tuple of time-to-expiry values (years).
    grid:
        ``grid[i_T][i_K]`` = local volatility (or *nan* where undefined).
        Shape is ``(len(maturities), len(strikes))``.
    """
    strikes: tuple[float, ...]
    maturities: tuple[float, ...]
    grid: tuple[tuple[float, ...], ...]


# ---------------------------------------------------------------------------
# Forward-curve interpolation (local helper)
# ---------------------------------------------------------------------------
def _forward_at(fs: FittedSurface, T: float) -> float:
    """Interpolate the forward price linearly in *T* from *forward_curve*.

    Uses ``numpy.interp`` (linear interpolation, flat extrapolation).
    """
    expiries = np.array([p[0] for p in fs.forward_curve])
    forwards = np.array([p[1] for p in fs.forward_curve])
    return float(np.interp(T, expiries, forwards))


# ---------------------------------------------------------------------------
# Finite-difference helpers
# ---------------------------------------------------------------------------
def _dw_dT(fs: FittedSurface, K: float, T: float,
           dT: float = _FD_T_DEFAULT) -> float:
    """First partial derivative of total variance w.r.t. time *T*.

    Central difference where possible; forward/backward at boundaries.
    """
    T_min = fs.fitted_slices[0].expiry_time
    T_max = fs.fitted_slices[-1].expiry_time

    # Near lower boundary — forward difference
    if T - dT < max(T_min, _T_MIN):
        wp = total_variance_at(fs, K, T + dT)
        w0 = total_variance_at(fs, K, T)
        return (wp - w0) / dT

    # Near upper boundary — backward difference
    if T + dT > T_max:
        w0 = total_variance_at(fs, K, T)
        wm = total_variance_at(fs, K, T - dT)
        return (w0 - wm) / dT

    # Interior — central difference
    wp = total_variance_at(fs, K, T + dT)
    wm = total_variance_at(fs, K, T - dT)
    return (wp - wm) / (2.0 * dT)


def _dw_dk(fs: FittedSurface, K: float, T: float, F_T: float,
           dK: float = _FD_K_DEFAULT) -> float:
    """First partial derivative of total variance w.r.t. log-moneyness *k*.

    Uses central difference in strike space, then converts the step to
    log-moneyness units.
    """
    # Very low / zero strike — forward difference
    K_min = dK  # effective minimum K for central diff
    if K - dK <= 0.0:
        kp = log((K + dK) / F_T)
        k0 = log(K / F_T)
        dk = kp - k0
        if abs(dk) < 1e-15:
            return nan
        wp = total_variance_at(fs, K + dK, T)
        w0 = total_variance_at(fs, K, T)
        return (wp - w0) / dk

    # Central difference
    dk = 0.5 * log((K + dK) / (K - dK))  # half the k-space interval
    if abs(dk) < 1e-15:
        return nan

    wp = total_variance_at(fs, K + dK, T)
    wm = total_variance_at(fs, K - dK, T)
    return (wp - wm) / (2.0 * dk)


def _d2w_dk2(fs: FittedSurface, K: float, T: float, F_T: float,
             dK: float = _FD_K_DEFAULT) -> float:
    """Second partial derivative of total variance w.r.t. log-moneyness *k*.

    Central difference formula: (w⁺ - 2·w₀ + w⁻) / dk².
    Returns *nan* if the strike step is too narrow for safe computation.
    """
    # Edge guard (can't do central second diff at boundary)
    if K - dK <= 0.0:
        return nan

    dk = 0.5 * log((K + dK) / (K - dK))
    if abs(dk) < 1e-15:
        return nan

    wp = total_variance_at(fs, K + dK, T)
    w0 = total_variance_at(fs, K, T)
    wm = total_variance_at(fs, K - dK, T)
    return (wp - 2.0 * w0 + wm) / (dk * dk)


# ---------------------------------------------------------------------------
# Dupire local volatility — single point
# ---------------------------------------------------------------------------
def dupire_at(fs: FittedSurface, K: float, T: float,
              dT: float = _FD_T_DEFAULT) -> float:
    """Dupire local volatility at a single (strike, time) point.

    Parameters
    ----------
    fs:
        Fitted volatility surface with at least one slice.
    K:
        Absolute strike (must be > 0).
    T:
        Time to expiry in years (must be within surface range).
    dT:
        Finite-difference step for the time derivative.

    Returns
    -------
    float
        Local volatility σ_loc(K, T), or *nan* if the Dupire formula
        denominator is non-positive (local volatility undefined).

    Raises
    ------
    ValueError
        If calendar arbitrage is detected (dw/dT ≤ ``_CAL_ARB_TOL``) or
        if *T* is outside the surface range (propagated from
        ``total_variance_at``).
    """
    w = total_variance_at(fs, K, T)

    # --- forward at this T ---
    F_T = _forward_at(fs, T)
    k = log(K / F_T)

    # --- time derivative ---
    dwdT = _dw_dT(fs, K, T, dT)
    if dwdT <= _CAL_ARB_TOL:
        raise ValueError(
            f"Calendar arbitrage at T={T:.6f}, K={K:.4f}: "
            f"dw/dT={dwdT:.8f} <= {_CAL_ARB_TOL}; "
            "Dupire local volatility undefined."
        )

    # --- moneyness derivatives ---
    # dK scales with strike at 0.1% relative (K * 1e-3); absolute floor
    # _FD_K_DEFAULT for very small strikes where relative step would underflow.
    dK = max(_FD_K_DEFAULT, K * _FD_K_DEFAULT)
    dwdk = _dw_dk(fs, K, T, F_T, dK)
    if isnan(dwdk):
        return nan

    d2wdk2 = _d2w_dk2(fs, K, T, F_T, dK)

    # --- Dupire denominator ---
    sqrt_w = sqrt(w)
    term2 = -(k / w) * dwdk
    term3 = 0.25 * (-1.0 / sqrt_w - 1.0 + (k * k) / w) * (dwdk * dwdk)
    term4 = 0.5 * d2wdk2
    denominator = 1.0 + term2 + term3 + term4

    if denominator <= _DENOM_MIN:
        return nan

    sigma_loc_sq = dwdT / denominator
    if sigma_loc_sq <= 0.0:
        return nan

    return sqrt(sigma_loc_sq)


# ---------------------------------------------------------------------------
# Dupire local volatility — full grid
# ---------------------------------------------------------------------------
def dupire(fs: FittedSurface,
           strikes: list[float],
           maturities: list[float],
           dT: float = _FD_T_DEFAULT) -> LocalVolSurface:
    """Build a :class:`LocalVolSurface` grid by calling ``dupire_at`` for
    every (K, T) pair.

    Parameters
    ----------
    fs:
        Fitted volatility surface.
    strikes:
        List of absolute strikes (``len >= 3``).
    maturities:
        List of time-to-expiry values in years (``len >= 3``).
    dT:
        Finite-difference step for the time derivative.

    Returns
    -------
    LocalVolSurface
        Frozen dataclass containing the grid.

    Raises
    ------
    ValueError
        If grid dimensions are too small.
    """
    if len(strikes) < 3:
        raise ValueError(
            f"Need at least 3 strikes, got {len(strikes)}"
        )
    if len(maturities) < 3:
        raise ValueError(
            f"Need at least 3 maturities, got {len(maturities)}"
        )

    grid: list[tuple[float, ...]] = []
    for T in maturities:
        row: list[float] = []
        for K in strikes:
            try:
                val = dupire_at(fs, K, T, dT)
            except ValueError:
                val = nan  # calendar-arb cell; mark undefined, don't abort
            row.append(val)
        grid.append(tuple(row))

    return LocalVolSurface(
        strikes=tuple(strikes),
        maturities=tuple(maturities),
        grid=tuple(grid),
    )
