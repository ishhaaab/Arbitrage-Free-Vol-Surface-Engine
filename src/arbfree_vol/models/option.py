from enum import Enum
from datetime import date
from pydantic import BaseModel, Field

class OptionType(str, Enum):
    CALL= "call"
    PUT= "put"

class OptionContract(BaseModel):
    symbol: str= Field(..., min_length=1)
    option_type: OptionType
    strike: float= Field(..., gt=0)
    expiry: date


class BlackScholesInput(BaseModel):
    contract: OptionContract
    spot: float= Field(..., gt=0)
    expiry_time: float= Field(..., gt=0) # time to expiry in years
    risk_free: float
    div_yield: float
    volatility: float= Field(..., gt=0)

