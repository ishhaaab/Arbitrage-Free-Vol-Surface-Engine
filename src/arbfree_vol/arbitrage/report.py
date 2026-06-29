from enum import Enum
from dataclasses import dataclass


class ViolationType(str, Enum):
    MONOTONICITY = "monotonicity"
    BUTTERFLY = "butterfly"
    CALENDAR = "calendar"
    PARITY = "parity"

@dataclass(frozen=True, slots=True)
class ArbitrageViolation:   
    kind: ViolationType
    detail: str                 #  a readable description, 
                                #  e.g. "call price increased from 5.2 to 5.8 at strikes 100 to 110"
    magnitude: float


@dataclass(frozen=True, slots=True)
class ArbitrageReport:
    violations: list[ArbitrageViolation]

    @property
    def is_arbitrage_free(self)-> bool:
        return len(self.violations) == 0