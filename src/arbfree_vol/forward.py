"""Per-expiry forward estimation from put-call parity, shared by the
arbitrage and repair layers.

``estimate_forward_curve`` extracts the forward price per expiry from
put-call parity (``F = e^{rT}(C-P) + K``, taking the median across
strikes), and ``populate_per_slice_r`` derives the per-slice risk-free
rate from the forward (``r = log(F/S)/T + q``).

This module is shared by ``arbitrage.quote_detect`` and
``repair.engine``.  It depends only on ``models`` — it has no
``repair`` dependency (the ``repair`` layer is a consumer, not a
requirement).
"""

import logging
from math import exp, log
from statistics import median

from arbfree_vol.models.surface import VolSurface, ExpirySlice, get_r, get_q
from arbfree_vol.models.option import OptionType

_logger = logging.getLogger(__name__)


def _slice_forward(s: ExpirySlice, r: float, spot: float) -> float | None:
    """Estimate forward price for one expiry slice via put call parity.

    Uses pairs of call/put at the same strike to solve for F from the
    put-call parity relation:

        C - P = e^{-rT} (F - K)

    Rearranged:  F = e^{rT} (C - P) + K

    Uses the **median** across strikes to prevent a single
    outlier quote from corrupting the estimate.  If no (call, put) pair
    exists, returns None (caller falls back).
    """
    by_strike: dict[float, dict[OptionType, float]] = {}
    for q in s.quotes:
        by_strike.setdefault(q.strike, {})[q.option_type] = q.price

    estimates: list[float] = []

    for K, sides in by_strike.items():
        if OptionType.CALL in sides and OptionType.PUT in sides:
            C = sides[OptionType.CALL]
            P = sides[OptionType.PUT]
            F_est = exp(r * s.expiry_time) * (C - P) + K
            if F_est > 0:
                estimates.append(F_est)

    if not estimates:
        return None

    return median(estimates)


def estimate_forward_curve(surface: VolSurface) -> dict[float, float]:
    """Estimate forward price per expiry from put call parity.

    For each slice, uses all available (call, put) pairs to extract
    the forward via C - P = e^{-rT} (F - K).  Returns a dict mapping
    expiry_time to forward_price.  Slices with zero pairs fall
    back to F = spot * exp((r - q) * T).

    The ``default substitution`` provenance suffix attached to the
    no-pair fallback warning is a HEURISTIC, not a provenance record:
    r/q matching the ingestion-layer default constants (0.05 / 0.0) is
    INFERRED to indicate a substituted (not observed) rate, but the same
    values can be genuinely observed.  The log wording says so explicitly
    ("this may be an observed value — provenance is inferred by this
    heuristic") and never asserts provenance.  Full provenance tracking
    (a flag recorded at the r/q source) is a larger, out-of-scope change.
    """
    spot = surface.spot
    curve: dict[float, float] = {}

    for s in surface.slices:
        r = get_r(surface, s)
        q = get_q(surface, s)
        F = _slice_forward(s, r, spot)
        if F is None:
            default_sub = abs(r - 0.05) < 1e-9 and abs(q) < 1e-9
            suffix = (
                " (r/q match the default substitution constants "
                "(0.05/0.0); this may be an observed value — provenance "
                "is inferred by this heuristic, not recorded)"
                if default_sub
                else " (r/q are non-default values)"
            )
            _logger.warning(
                "Slice T=%.4f has no (call, put) parity pair; forward "
                "falls back to theoretical spot*exp((r-q)*T) with "
                "r=%.4f, q=%.4f%s",
                s.expiry_time, r, q, suffix,
            )
            F = spot * exp((r - q) * s.expiry_time)
        curve[s.expiry_time] = F

    return curve


def populate_per_slice_r(surface: VolSurface, fwd_curve: dict[float, float]) -> None:
    """Set per-slice risk_free from the forward curve estimate.

    For each slice: r(T) = log(F / S) / T + q.  Uses the per-slice
    dividend yield (via ``get_q``) if set, otherwise falls back to
    the surface-level ``div_yield``.  Slices without a valid forward
    keep their current value (None = falls back to surface.r).

    The per-slice q pattern matches the per-slice-r pattern already
    in the codebase (``get_r`` / ``get_q`` in ``models/surface.py``).
    """
    for sl in surface.slices:
        F = fwd_curve.get(sl.expiry_time)
        if F is not None and F > 0 and sl.expiry_time > 0:
            q = get_q(surface, sl)  # per-slice q (falls back to surface-level)
            sl.risk_free = log(F / surface.spot) / sl.expiry_time + q
