from pydantic import BaseModel, Field

from arbfree_vol.models.option import OptionType

class Quote(BaseModel): # one option quote
    # price has no constraint so the ingestion layer can construct raw
    # quotes from messy market data; the cleaning layer is responsible
    # for rejecting price=0 and negative prices with an audit record.
    # After cleaning, kept quotes should have price > 0 which is enforced by
    # the cleaning rule and not by Pydantic.
    strike: float= Field(..., gt=0)
    option_type: OptionType
    price: float
    bid: float | None= None
    ask: float | None= None


class ExpirySlice(BaseModel):  # set of quotes with the same expiry
    expiry_time: float= Field(..., gt=0)
    quotes: list[Quote]= Field(..., min_length=1)
    risk_free: float | None= None   # per-slice override for surface-level risk_free
    div_yield: float | None= None   # per-slice override for surface-level div_yield

class VolSurface(BaseModel): # set of slices of different expirations
    spot: float= Field(..., gt=0)
    risk_free: float
    div_yield: float
    slices: list[ExpirySlice]= Field(..., min_length=1)


def get_r(surface: VolSurface, sl: ExpirySlice) -> float:
    """Return the effective risk-free rate for a slice.

    Prefers the per-slice value (if set), falls back to the surface-level
    default.  This lets us store per-expiry term structure without
    changing every call site.
    """
    return sl.risk_free if sl.risk_free is not None else surface.risk_free


def get_q(surface: VolSurface, sl: ExpirySlice) -> float:
    """Return the effective dividend yield for a slice."""
    return sl.div_yield if sl.div_yield is not None else surface.div_yield
