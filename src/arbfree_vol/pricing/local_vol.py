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
           + \\frac14\\bigl(-\\frac14 - \\frac1{w} + \\frac{k^2}{w^2}\\bigr)
             (\\partial_k w)^2
           + \\frac12 \\partial_{kk} w }

where :math:`k = \\ln(K / F(T))` and all partial derivatives are
approximated with finite differences.  See ``_dupire_denominator`` for
the denominator terms, ``_dw_dT`` for the time derivative (fixed-``k``
re-striking), and ``_d2w_dk2`` for the non-uniform-grid second
derivative — those docstrings carry the numerical caveats.

Fallback-slice contamination handling (``_stencil_touches_fallback``,
``_FB_TOL``) lives in ``_fallback``; finite-difference step sizes are
module-level constants below.
"""

from dataclasses import dataclass
from math import log, nan, sqrt, isnan

from arbfree_vol.surface.interpolate import FittedSurface, total_variance_at, _forward_at
from arbfree_vol.pricing._fallback import _FB_TOL, _stencil_touches_fallback


# Module-level tolerances  

_FD_T_DEFAULT: float = 1e-3       # default finite-difference step in T (years)
_FD_K_DEFAULT: float = 1e-3       # default absolute strike step (used when
                                  # caller does not pass dK explicitly)
_DENOM_MIN: float = 1e-10         # denominator values ≤ this => local-vol
                                  # undefined (return nan)
_CAL_ARB_TOL: float = 0.0         # dw/dT ≤ tol => calendar arbitrage => raise
_T_MIN: float = 1e-4              # absolute tiny threshold for T near zero


# LocalVolSurface to compute boundary type  
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



# Finite-difference helpers
def _dw_dT(fs: FittedSurface, K: float, T: float,
           dT: float = _FD_T_DEFAULT) -> float:
    """First partial derivative of total variance w.r.t. time *T*.

    Central difference where possible; forward/backward at boundaries.

    The derivative is taken at FIXED log-moneyness ``k = ln(K/F(T))``,
    as required by the Gatheral (2004) Eq 1.10 form of the Dupire
    formula.  Because ``total_variance_at`` interpolates at a fixed
    *absolute strike*, the stencil points must re-strike along the
    forward curve: ``K' = K * F(T±dT) / F(T)``.  Holding *K* fixed
    instead injects a spurious ``(r-q) * dw/dk`` term into the numerator
    (``k`` drifts by ``-d ln F/dT * dT`` across the stencil); with a
    non-zero carry and a skewed smile that bias is first-order in the
    slope and can reach several percent of the local vol.
    """
    T_min = fs.fitted_slices[0].expiry_time
    T_max = fs.fitted_slices[-1].expiry_time

    F_T = _forward_at(fs, T)

    # Near lower boundary — forward difference
    if T - dT < max(T_min, _T_MIN):
        K_p = K * _forward_at(fs, T + dT) / F_T
        wp = total_variance_at(fs, K_p, T + dT)
        w0 = total_variance_at(fs, K, T)
        return (wp - w0) / dT

    # Near upper boundary — backward difference
    if T + dT > T_max:
        K_m = K * _forward_at(fs, T - dT) / F_T
        w0 = total_variance_at(fs, K, T)
        wm = total_variance_at(fs, K_m, T - dT)
        return (w0 - wm) / dT

    # Interior — central difference
    K_p = K * _forward_at(fs, T + dT) / F_T
    K_m = K * _forward_at(fs, T - dT) / F_T
    wp = total_variance_at(fs, K_p, T + dT)
    wm = total_variance_at(fs, K_m, T - dT)
    return (wp - wm) / (2.0 * dT)


def _dw_dk(fs: FittedSurface, K: float, T: float, F_T: float,
           dK: float = _FD_K_DEFAULT) -> float:
    """First partial derivative of total variance w.r.t. log-moneyness *k*.

    Uses central difference in strike space, then converts the step to
    log-moneyness units.
    """
    # Very low / zero strike: forward difference
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

    Correct central second difference on the NON-UNIFORM k-grid produced
    by equal absolute-strike steps (``k = ln(K/F)`` makes equal K-steps
    give unequal k-steps).  With ``h⁺ = k⁺ - k⁰`` and ``h⁻ = k⁰ - k⁻``:

    .. math::

        w''(k_0) = \\frac{2}{h^+ + h^-}
        \\left[ \\frac{w^+ - w^0}{h^+} - \\frac{w^0 - w^-}{h^-} \\right]

    On a uniform grid (``h⁺ = h⁻ = h``) this reduces exactly to the
    symmetric formula ``(w⁺ - 2·w⁰ + w⁻) / h²``.  The symmetric formula
    applied to the asymmetric k-grid would inject a spurious ``-w'(k)``
    first-derivative term (measured d²w = -b on a linear branch where the
    true d²w = 0); this non-uniform stencil removes that bias.
    Returns *nan* if the strike step is too narrow for safe computation.
    """
    # Edge guard (can't do central second diff at boundary)
    if K - dK <= 0.0:
        return nan

    # k-space half-steps around the center point (F_T cancels in each ratio)
    h_plus = log((K + dK) / K)
    h_minus = log(K / (K - dK))
    if abs(h_plus) < 1e-15 or abs(h_minus) < 1e-15:
        return nan

    wp = total_variance_at(fs, K + dK, T)
    w0 = total_variance_at(fs, K, T)
    wm = total_variance_at(fs, K - dK, T)
    return 2.0 / (h_plus + h_minus) * (
        (wp - w0) / h_plus - (w0 - wm) / h_minus
    )


def _dupire_denominator(
    w: float,
    k: float,
    dwdk: float,
    d2wdk2: float,
) -> float:
    """Gatheral (2004) eq. 1.10 denominator from total-variance derivatives.

    ``denominator = 1 - (k/w)·w' + ¼·(-¼ - 1/w + k²/w²)·(w')² + ½·w''``
    where ``w`` is total variance, ``k`` is log-moneyness, and the
    derivatives are w.r.t. ``k``.  A non-positive denominator means local
    volatility is undefined at this point — the caller maps that to *nan*.
    """
    term2 = -(k / w) * dwdk
    term3 = 0.25 * (-0.25 - 1.0 / w + (k * k) / (w * w)) * (dwdk * dwdk)
    term4 = 0.5 * d2wdk2
    return 1.0 + term2 + term3 + term4


# Dupire local volatility (single point)
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

    # moneyness derivatives:
    # dK scales with strike at 0.1% relative (K * 1e-3); absolute floor
    # _FD_K_DEFAULT for very small strikes where relative step would underflow.
    dK = max(_FD_K_DEFAULT, K * _FD_K_DEFAULT)
    dwdk = _dw_dk(fs, K, T, F_T, dK)
    if isnan(dwdk):
        return nan

    d2wdk2 = _d2w_dk2(fs, K, T, F_T, dK)

    # Dupire denominator
    denominator = _dupire_denominator(w, k, dwdk, d2wdk2)

    if denominator <= _DENOM_MIN:
        return nan

    sigma_loc_sq = dwdT / denominator
    if sigma_loc_sq <= 0.0:
        return nan

    return sqrt(sigma_loc_sq)



# Dupire local volatility (full grid)

def dupire(fs: FittedSurface,
           strikes: list[float],
           maturities: list[float],
           dT: float = _FD_T_DEFAULT,
           fallback_slices: list[float] | None = None) -> LocalVolSurface:
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
    fallback_slices:
        Optional list of T values that used the unconstrained fallback
        path during eSSVI sequential fitting.  When provided, any grid
        row whose FD stencil (T-dT, T, T+dT) reaches into an
        interpolation interval that borders a fallback slice is set to
        *nan*.  This prevents derivative leakage from contaminated
        interpolation regions.

    Returns
    -------
    LocalVolSurface
        Frozen dataclass containing the grid.

    Raises
    ------
    ValueError
        If grid dimensions are too small, or if the fitted surface has
        fewer than two slices and any grid row would actually be
        evaluated (a sub-2-slice surface cannot produce a Dupire time
        derivative; the only such surface that is accepted is one whose
        every grid row is masked as a fallback maturity).
    """
    if len(strikes) < 3:
        raise ValueError(
            f"Need at least 3 strikes, got {len(strikes)}"
        )
    if len(maturities) < 3:
        raise ValueError(
            f"Need at least 3 maturities, got {len(maturities)}"
        )

    # Pre-compute fallback contamination lookup
    fallback_set, fitted_times = _fallback_precompute(fs, fallback_slices)

    # A sub-2-slice surface cannot produce a Dupire time derivative: the
    # interpolation path (total_variance_at / _dw_dT) needs an interior
    # interval to bracket, and a non-fallback grid row leaks an obscure
    # out-of-range ValueError.  The only legitimate sub-2-slice grid is
    # one whose EVERY row is masked as a fallback maturity (all-NaN
    # without evaluation) — any other row must fail clearly here.
    if len(fs.fitted_slices) < 2:
        every_row_masked = bool(fallback_set) and all(
            any(abs(T - fb) < _FB_TOL for fb in fallback_set)
            for T in maturities
        )
        if not every_row_masked:
            raise ValueError(
                f"dupire requires at least 2 fitted slices; got "
                f"{len(fs.fitted_slices)}"
            )

    grid: list[tuple[float, ...]] = []
    for T in maturities:
        # Check if this row's stencil is contaminated by a fallback slice
        if fallback_slices and _stencil_touches_fallback(
            T, fitted_times, fallback_set, dT
        ):
            grid.append(tuple(nan for _ in strikes))
            continue

        grid.append(tuple(_eval_cell(fs, K, T, dT) for K in strikes))

    return LocalVolSurface(
        strikes=tuple(strikes),
        maturities=tuple(maturities),
        grid=tuple(grid),
    )


def _fallback_precompute(
    fs: FittedSurface,
    fallback_slices: list[float] | None,
) -> tuple[set[float], tuple[float, ...]]:
    """Build the fallback-contamination lookup context for ``dupire``.

    Returns ``(fallback_set, fitted_times)`` where ``fallback_set`` is the
    fallback T values (empty when ``fallback_slices`` is falsy) and
    ``fitted_times`` is the sorted tuple of fitted-slice expiries used to
    bracket the FD stencil.
    """
    if not fallback_slices:
        return set(), ()
    return (
        set(fallback_slices),
        tuple(sorted(s.expiry_time for s in fs.fitted_slices)),
    )


def _eval_cell(fs: FittedSurface, K: float, T: float, dT: float) -> float:
    """Evaluate Dupire local vol at one grid cell.

    A calendar-arbitrage ``ValueError`` (from ``dupire_at``) is mapped to
    *nan* — the cell is marked undefined, not fatal.  Genuine out-of-range
    errors (missing slice, no slices, strike below/above the surface) are
    re-raised.
    """
    try:
        return dupire_at(fs, K, T, dT)
    except ValueError as e:
        msg = str(e)
        if "not found" in msg or "no slices" in msg or "below" in msg or "above" in msg:
            raise
        return nan  # calendar-arb cell; mark undefined, don't abort
