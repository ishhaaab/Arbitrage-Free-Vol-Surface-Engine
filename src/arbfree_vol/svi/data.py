from math import exp

from arbfree_vol.models.surface import VolSurface, ExpirySlice, get_r, get_q


def _forward_price(surface: VolSurface, s: ExpirySlice) -> float:
    """Forward price F = S * e^{(r - q)T}."""
    r = get_r(surface, s)
    q = get_q(surface, s)
    return surface.spot * exp((r - q) * s.expiry_time)
