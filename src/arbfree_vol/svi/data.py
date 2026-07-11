from math import exp, log

from arbfree_vol.models.surface import VolSurface, ExpirySlice
from arbfree_vol.variance import slice_total_variance


def _forward_price(surface: VolSurface, s: ExpirySlice) -> float:
    """Forward price F = S * e^{(r - q)T}."""
    return surface.spot * exp((surface.risk_free - surface.div_yield) * s.expiry_time)


def slice_to_point(surface: VolSurface, s: ExpirySlice) -> list[tuple[float, float]]:
    """Convert a slice to (k, w) points: k = ln(K / F), w = total variance."""
    strike_w = slice_total_variance(surface, s)
    F = _forward_price(surface, s)

    points = [(log(strike / F), w) for strike, w in strike_w.items()]
    return sorted(points)
