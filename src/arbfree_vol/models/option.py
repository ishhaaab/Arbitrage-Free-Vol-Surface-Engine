from enum import Enum
from datetime import date
from pydantic import BaseModel, Field
from dataclasses import dataclass

class OptionType(str, Enum):
    CALL= "call"
    PUT= "put"


@dataclass(frozen=True, slots=True)
class OffendingQuote:
    """Structured reference to a quote flagged by an arb check.

    Used by the repair engine to map a violation back to the specific
    quote(s) that should be rejected.  Lives in the kernel because
    both `arbitrage` and `repair` need it.
    """
    strike: float
    expiry_time: float
    option_type: OptionType

class OptionContract(BaseModel):
    symbol: str= Field(..., min_length=1)
    option_type: OptionType
    strike: float= Field(..., gt=0)
    expiry_date: date


class BlackScholesInput(BaseModel):
    contract: OptionContract
    spot: float= Field(..., gt=0)
    expiry_time: float= Field(..., gt=0) # time to expiry in years
    risk_free: float
    div_yield: float
    volatility: float= Field(..., gt=0)


class ImpliedVolInput(BaseModel):
    contract: OptionContract
    spot: float= Field(..., gt=0)
    expiry_time: float= Field(..., gt=0) # time to expiry in years
    risk_free: float
    div_yield: float
    market_price: float= Field(..., gt=0)
