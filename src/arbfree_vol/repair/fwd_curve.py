from math import exp, log
from statistics import median

from arbfree_vol.models.surface import VolSurface, ExpirySlice, get_r
from arbfree_vol.models.option import OptionType


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
    Ks_used: list[float] = []

    for K, sides in by_strike.items():
        if OptionType.CALL in sides and OptionType.PUT in sides:
            C = sides[OptionType.CALL]
            P = sides[OptionType.PUT]
            F_est = exp(r * s.expiry_time) * (C - P) + K
            estimates.append(F_est)
            Ks_used.append(K)

    if not estimates:
        return None

    return median(estimates)


def estimate_forward_curve(surface: VolSurface) -> dict[float, float]:
    """Estimate forward price per expiry from put call parity.

    For each slice, uses all available (call, put) pairs to extract
    the forward via C - P = e^{-rT} (F - K).  Returns a dict mapping
    expiry_time to forward_price.  Slices with zero pairs fall
    back to F = spot * exp(r * T) due to q = 0 assumption.
    """
    spot = surface.spot
    curve: dict[float, float] = {}

    for s in surface.slices:
        r = get_r(surface, s)
        F = _slice_forward(s, r, spot)
        if F is None:
            F = spot * exp(r * s.expiry_time)  # q = 0 fallback
        curve[s.expiry_time] = F

    return curve


def populate_per_slice_r(surface: VolSurface, fwd_curve: dict[float, float]) -> None:
    """Compute per-maturity risk-free rates from the forward curve.

    For each slice, solves  ``F(T) = S * exp((r - q) * T)`` for ``r``::

        r(T) = log(F(T) / S) / T + q

    and stores the result on ``sl.risk_free``.  Slices without a valid
    forward estimate keep their current value (typically ``None``, which
    falls back to ``surface.risk_free`` via ``get_r()``).
    """
    q = surface.div_yield
    for sl in surface.slices:
        F = fwd_curve.get(sl.expiry_time)
        if F is not None and F > 0 and sl.expiry_time > 0:
            sl.risk_free = log(F / surface.spot) / sl.expiry_time + q
