from pydantic import BaseModel, Field

from arbfree_vol.models.option import OptionType

class Quote(BaseModel): # one option quote 
    strike: float= Field(..., gt=0)
    option_type: OptionType
    price: float= Field(..., gt=0)
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
