"""Per-expiry forward estimation from put-call parity."""

from math import exp
from statistics import median

from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import ExpirySlice, VolSurface, get_q, get_r


def _slice_forward(expiry_slice: ExpirySlice, risk_free: float) -> float | None:
    """Return the median strike-level parity estimate for one expiry."""
    by_strike: dict[float, dict[OptionType, float]] = {}
    for quote in expiry_slice.quotes:
        by_strike.setdefault(quote.strike, {})[quote.option_type] = quote.price

    estimates = [
        exp(risk_free * expiry_slice.expiry_time)
        * (sides[OptionType.CALL] - sides[OptionType.PUT])
        + strike
        for strike, sides in by_strike.items()
        if OptionType.CALL in sides and OptionType.PUT in sides
    ]
    positive = [estimate for estimate in estimates if estimate > 0]
    return median(positive) if positive else None


def estimate_forward_curve(surface: VolSurface) -> dict[float, float]:
    """Estimate each forward from parity, with a documented carry fallback."""
    curve: dict[float, float] = {}
    for expiry_slice in surface.slices:
        risk_free = get_r(surface, expiry_slice)
        dividend_yield = get_q(surface, expiry_slice)
        forward = _slice_forward(expiry_slice, risk_free)
        if forward is None:
            forward = surface.spot * exp(
                (risk_free - dividend_yield) * expiry_slice.expiry_time
            )
        curve[expiry_slice.expiry_time] = forward
    return curve
