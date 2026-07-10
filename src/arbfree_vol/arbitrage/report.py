from enum import Enum
from dataclasses import dataclass, field
from arbfree_vol.models.option import OffendingQuote


class ViolationType(str, Enum):
    MONOTONICITY= "monotonicity"
    BUTTERFLY= "butterfly"
    CALENDAR= "calendar"
    PARITY= "parity"
    NEGATIVE_VARIANCE= "negative_variance"
    WIDE_SPREAD= "wide_bid_ask"


@dataclass(frozen=True, slots=True)
class ArbitrageViolation:
    kind: ViolationType
    detail: str                 #  a readable description,
                                #  e.g. "call price increased from 5.2 to 5.8 at strikes 100 to 110"
    magnitude: float
    # structured refs to the offending quote(s).  Empty for curve shape
    # violations (NEGATIVE_VARIANCE) where no single quote is to blame.
    # A tuple keeps this hashable and immutable.
    offending: tuple[OffendingQuote, ...] = ()




@dataclass(frozen=True, slots=True)
class ArbitrageReport:
    violations: list[ArbitrageViolation]

    @property
    def is_arbitrage_free(self)-> bool:
        return len(self.violations) == 0