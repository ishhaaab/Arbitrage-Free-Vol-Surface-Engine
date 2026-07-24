"""Fitted surface interpolation routines.

Provides a frozen dataclass ``FittedSurface`` that holds the calibrated
smile parameters for a set of expiries together with the forward curve,
and routines to interpolate total variance / Black-Scholes implied
volatility at arbitrary strikes and expiries.
"""

from dataclasses import dataclass
from math import log, sqrt

import numpy as np

from arbfree_vol.repair.report import FittedSlice, RepairReport
from arbfree_vol.svi.model import svi_total_variance

# ---------------------------------------------------------------------------
# Module-level tolerance constants (no-hardcoding rule)
# ---------------------------------------------------------------------------
_EXACT_EXPIRY_TOL: float = 1e-10


# ---------------------------------------------------------------------------
# FittedSurface — compute boundary type (frozen dataclass, no Pydantic)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FittedSurface:
    """Stripped-down fitted vol surface for analytics.

    All three smile-model code paths (SVI / eSSVI / SABR) funnel their
    fitted parameters through raw SVI ``FittedSlice`` objects, so
    ``FittedSurface`` works uniformly regardless of which model was used
    during repair.
    """

    spot: float
    risk_free: float
    div_yield: float

    forward_curve: tuple[tuple[float, float], ...]
    """(expiry_time, forward_price) pairs sorted ascending by expiry."""

    fitted_slices: tuple[FittedSlice, ...]
    """Fitted slices sorted ascending by expiry_time."""


def build_fitted_surface(report: RepairReport) -> FittedSurface:
    """Construct a ``FittedSurface`` from a completed ``RepairReport``.

    Parameters
    ----------
    report:
        Output of ``repair()`` containing cleaned surface and fitted
        SVI slices.

    Returns
    -------
    FittedSurface
        Ready for interpolation and risk analytics.

    Raises
    ------
    ValueError
        If ``report.cleaned_surface`` is ``None`` (no valid surface
        survived the repair process).
    """
    cleaned = report.cleaned_surface
    if cleaned is None:
        raise ValueError(
            "RepairReport has no cleaned_surface; cannot build FittedSurface"
        )

    spot = cleaned.spot
    risk_free = cleaned.risk_free
    div_yield = cleaned.div_yield

    # Build forward curve from fitted slices (each carries its own forward).
    fwd_tuples: list[tuple[float, float]] = [
        (fs.expiry_time, fs.forward_price) for fs in report.fitted_slices
    ]
    fwd_tuples.sort(key=lambda x: x[0])

    sorted_slices = tuple(sorted(report.fitted_slices, key=lambda fs: fs.expiry_time))

    return FittedSurface(
        spot=spot,
        risk_free=risk_free,
        div_yield=div_yield,
        forward_curve=tuple(fwd_tuples),
        fitted_slices=sorted_slices,
    )


# ---------------------------------------------------------------------------
# Interpolation helpers
# ---------------------------------------------------------------------------
def _forward_at(fs: FittedSurface, T: float) -> float:
    """Interpolate the forward price linearly in *T* from *forward_curve*.

    Uses ``numpy.interp`` (linear interpolation, flat extrapolation).
    Callers must validate *T* against the surface range before calling
    (done by ``total_variance_at``).
    """
    expiries = np.array([p[0] for p in fs.forward_curve])
    forwards = np.array([p[1] for p in fs.forward_curve])
    return float(np.interp(T, expiries, forwards))


def total_variance_at(fs: FittedSurface, K: float, T: float) -> float:
    """Total variance *w(K, T)* interpolated from the fitted surface.

    Parameters
    ----------
    fs:
        Fitted surface with at least one slice.
    K:
        Absolute strike.
    T:
        Time to expiry in years.

    Returns
    -------
    float
        Total variance *σ²·T*.

    Raises
    ------
    ValueError
        If *T* is outside the bracketed range of the surface.
    """
    slices = fs.fitted_slices
    n = len(slices)

    if n == 0:
        raise ValueError("FittedSurface has no slices; cannot interpolate")

    T_min = slices[0].expiry_time
    T_max = slices[-1].expiry_time

    if T < T_min - _EXACT_EXPIRY_TOL:
        raise ValueError(
            f"T={T} is below the surface range [{T_min}, {T_max}]"
        )
    if T > T_max + _EXACT_EXPIRY_TOL:
        raise ValueError(
            f"T={T} is above the surface range [{T_min}, {T_max}]"
        )

    # ── exact slice expiry match ──────────────────────────────────────────
    for i, sl in enumerate(slices):
        if abs(sl.expiry_time - T) < _EXACT_EXPIRY_TOL:
            F = _forward_at(fs, sl.expiry_time)
            k = log(K / F)
            w = svi_total_variance(
                k, sl.params.a, sl.params.b, sl.params.rho,
                sl.params.m, sl.params.sigma
            )
            return w

    # ── interior interpolation ────────────────────────────────────────────
    # Locate the bracketing pair.
    i_low = 0
    i_high = n - 1
    for i in range(n - 1):
        if slices[i].expiry_time <= T <= slices[i + 1].expiry_time:
            i_low = i
            i_high = i + 1
            break

    sl_low = slices[i_low]
    sl_high = slices[i_high]

    T_low = sl_low.expiry_time
    T_high = sl_high.expiry_time

    F_low = _forward_at(fs, T_low)
    F_high = _forward_at(fs, T_high)

    k_low = log(K / F_low)
    k_high = log(K / F_high)

    w_low = svi_total_variance(
        k_low, sl_low.params.a, sl_low.params.b, sl_low.params.rho,
        sl_low.params.m, sl_low.params.sigma
    )
    w_high = svi_total_variance(
        k_high, sl_high.params.a, sl_high.params.b, sl_high.params.rho,
        sl_high.params.m, sl_high.params.sigma
    )

    # Linear interpolation in total-variance space.
    if abs(T_high - T_low) < _EXACT_EXPIRY_TOL:
        # Fallback (shouldn't be reached given exact-match path above).
        return w_low

    theta = (T - T_low) / (T_high - T_low)
    w = w_low + theta * (w_high - w_low)
    return w


def iv_at(fs: FittedSurface, K: float, T: float) -> float:
    """Black-Scholes implied volatility from the fitted surface.

    Equivalent to ``sqrt(total_variance_at(fs, K, T) / T)``.

    Parameters
    ----------
    fs:
        Fitted surface.
    K:
        Absolute strike.
    T:
        Time to expiry in years.

    Returns
    -------
    float
        Annualised implied volatility.
    """
    w = total_variance_at(fs, K, T)
    return sqrt(w / T)
