from math import exp

from arbfree_vol.models.surface import VolSurface, ExpirySlice
from arbfree_vol.models.option import OptionType


def _slice_forward(s: ExpirySlice, r: float, spot: float) -> float | None:
    """Estimate forward price for one expiry slice via put call parity.

    Uses pairs of call/put at the same strike to solve for F from the
    put-call parity relation:

        C - P = e^{-rT} (F - K)

    Rearranged:  F = e^{rT} (C - P) + K

    If no (call, put) pair exists, returns None (caller falls back).
    If multiple pairs exist, returns the arithmetic mean.
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

    return sum(estimates) / len(estimates)


def estimate_forward_curve(surface: VolSurface) -> dict[float, float]:
    """Estimate forward price per expiry from put call parity.

    For each slice, uses all available (call, put) pairs to extract
    the forward via C - P = e^{-rT} (F - K).  Returns a dict mapping
    expiry_time to forward_price.  Slices with zero pairs fall
    back to F = spot * exp(r * T) due to q = 0 assumption.
    """
    r = surface.risk_free
    spot = surface.spot
    curve: dict[float, float] = {}

    for s in surface.slices:
        F = _slice_forward(s, r, spot)
        if F is None:
            F = spot * exp(r * s.expiry_time)  # q = 0 fallback
        curve[s.expiry_time] = F

    return curve
