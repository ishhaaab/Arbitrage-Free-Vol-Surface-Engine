"""Calendar-spread (total-variance monotonicity) arbitrage detection.

Self-contained piece of the quote-level detection layer: converts each
expiry slice to ``(log-moneyness, total-variance)`` space and flags
contiguous ``k``-bands where an earlier slice's total variance exceeds a
later slice's (calendar arbitrage).

``slice_total_variance`` is imported into this module's namespace so the
regression tests can monkeypatch it here (``tests/test_arbitrage.py``
patches ``arbitrage.calendar.slice_total_variance`` to simulate a slice
with empty total variance).
"""

import numpy as np
from math import log

from arbfree_vol.arbitrage.report import ArbitrageViolation, ViolationType
from arbfree_vol.models.surface import VolSurface, ExpirySlice
from arbfree_vol.variance import slice_total_variance


def _check_calendar(surface: VolSurface,
                    violations: list[ArbitrageViolation],
                    n_k: int = 61) -> None:
    """Total variance must be non-decreasing with time at every log-moneyness k.

    Converts each slice to (k, w) space using its per-slice forward price,
    then interpolates onto a common k-grid per adjacent pair.  Flags
    contiguous bands where w_earlier(k) > w_later(k) beyond tolerance.
    """
    ordered = sorted(surface.slices, key=lambda sl: sl.expiry_time)
    tolerance = 1e-4

    for i in range(len(ordered) - 1):
        earlier = ordered[i]
        later = ordered[i + 1]

        grids = _calendar_pair_grids(surface, earlier, later, n_k)
        if grids is None:
            continue

        k_grid, gap = grids
        _append_calendar_violations(
            violations, earlier, later, k_grid, gap, tolerance)


def _calendar_pair_grids(
    surface: VolSurface,
    earlier: ExpirySlice,
    later: ExpirySlice,
    n_k: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Interpolate both slices' total variance onto a common k-grid.

    Returns ``(k_grid, gap)`` where ``gap = w_earlier - w_later`` evaluated
    on the shared log-moneyness grid, or ``None`` when the pair has no
    usable overlap (fewer than 2 points on either side, or disjoint k
    ranges).
    """
    from arbfree_vol.svi.data import _forward_price

    F_e = _forward_price(surface, earlier)
    F_l = _forward_price(surface, later)

    w_e = slice_total_variance(surface, earlier)
    w_l = slice_total_variance(surface, later)

    ew = sorted([(log(K / F_e), w) for K, w in w_e.items()])
    lw = sorted([(log(K / F_l), w) for K, w in w_l.items()])

    if len(ew) < 2 or len(lw) < 2:
        return None

    ks_e, vs_e = zip(*ew)
    ks_l, vs_l = zip(*lw)

    k_min = max(min(ks_e), min(ks_l))
    k_max = min(max(ks_e), max(ks_l))
    if k_min >= k_max:
        return None

    k_grid = np.linspace(k_min, k_max, n_k)
    w_e_interp = np.interp(k_grid, ks_e, vs_e)
    w_l_interp = np.interp(k_grid, ks_l, vs_l)
    return k_grid, w_e_interp - w_l_interp


def _append_calendar_violations(
    violations: list[ArbitrageViolation],
    earlier: ExpirySlice,
    later: ExpirySlice,
    k_grid: np.ndarray,
    gap: np.ndarray,
    tolerance: float,
) -> None:
    """Flag contiguous runs where ``gap > tolerance`` as calendar arbitrage.

    A run is a maximal set of adjacent grid points with ``gap > tolerance``;
    each run becomes one CALENDAR violation naming the pair's expiries, the
    run's k-range, and its worst gap.
    """
    in_run = False
    run_start = 0
    max_gap = 0.0

    for j in range(len(k_grid)):
        if gap[j] > tolerance:
            if not in_run:
                run_start = j
                max_gap = gap[j]
                in_run = True
            else:
                max_gap = max(max_gap, gap[j])
        else:
            if in_run:
                violations.append(_calendar_violation(
                    earlier, later, k_grid, run_start, j - 1, max_gap))
                in_run = False

    if in_run:
        violations.append(_calendar_violation(
            earlier, later, k_grid, run_start, len(k_grid) - 1, max_gap))


def _calendar_violation(
    earlier: ExpirySlice,
    later: ExpirySlice,
    k_grid: np.ndarray,
    i_start: int,
    i_end: int,
    max_gap: float,
) -> ArbitrageViolation:
    """Build one CALENDAR violation for a contiguous run ``[i_start, i_end]``."""
    return ArbitrageViolation(
        kind=ViolationType.CALENDAR,
        detail=f"calendar arb: T={earlier.expiry_time:.4f} > T={later.expiry_time:.4f}, "
                f"k=[{k_grid[i_start]:.4f}, {k_grid[i_end]:.4f}], "
                f"worst gap={max_gap:.6f}",
        magnitude=max_gap,
        offending=(),
    )
