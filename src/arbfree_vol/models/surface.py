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

class VolSurface(BaseModel): # set of slices of different expirations
    spot: float= Field(..., gt=0)
    risk_free: float
    div_yield: float
    slices: list[ExpirySlice]= Field(..., min_length=1)
