from dataclasses import dataclass, field
from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import VolSurface
from arbfree_vol.models.fitted import FittedSlice, FittedSSVISlice, FittedSABRSlice
from arbfree_vol.arbitrage.report import ViolationType, ArbitrageReport



@dataclass(frozen=True, slots=True)
class RejectedQuote:

    strike: float
    expiry_time: float
    option_type: OptionType
    reason: ViolationType
    magnitude: float

@dataclass(frozen=True, slots=True)
class RepairMetrics:

    n_rejected: int
    n_total_quotes: int
    n_slices_input: int
    n_slices_fitted: int
    n_violations_before: int
    n_violations_after: int

    @property
    def rejection_rate(self) -> float:
        return self.n_rejected / self.n_total_quotes if self.n_total_quotes > 0 else 0.0

@dataclass(frozen=True, slots=True)
class RepairReport:

    rejected: tuple[RejectedQuote, ...]
    fitted_slices: tuple[FittedSlice, ...]
    remaining_violations: ArbitrageReport
    metrics: RepairMetrics
    cleaned_surface: VolSurface | None
    fitted_ssvi_slices: tuple[FittedSSVISlice, ...] = ()
    fitted_sabr_slices: tuple[FittedSABRSlice, ...] = ()
    repair_infeasible: bool = False
    fallback_slices: list[float] = field(default_factory=list)
    failed_slices: list[float] = field(default_factory=list)
    # Slices whose SABR->SVI mapping raised RuntimeError; they are NOT in
    # fitted_sabr_slices/fitted_slices; repair_infeasible semantics unchanged.
    sabr_mapping_failed_slices: list[float] = field(default_factory=list)
