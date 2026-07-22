
from dataclasses import dataclass
from arbfree_vol.models.option import OptionType, OffendingQuote
from arbfree_vol.models.surface import VolSurface
from arbfree_vol.arbitrage.report import ViolationType, ArbitrageReport
from arbfree_vol.svi.model import SVIParams
from arbfree_vol.ssvi.model import SSVIParams, eSSVISurfaceParams
from arbfree_vol.sabr.model import SABRParams



@dataclass(frozen=True, slots=True)
class RejectedQuote:

    strike: float
    expiry_time: float
    option_type: OptionType
    reason: ViolationType
    magnitude: float

@dataclass(frozen=True, slots=True)
class FittedSlice:

    expiry_time: float
    params: SVIParams
    rmse: float
    forward_price: float
    n_quotes_total: int
    n_quotes_used: int
    data_points: tuple[tuple[float, float], ...] | None= None

@dataclass(frozen=True, slots=True)
class FittedSSVISlice:
    """SSVI fit for one slice, with optional eSSVI surface parameters."""
    expiry_time: float
    ssvi: SSVIParams
    rmse: float
    forward_price: float
    n_quotes_total: int
    n_quotes_used: int
    essvi: eSSVISurfaceParams | None = None

@dataclass(frozen=True, slots=True)
class FittedSABRSlice:
    """SABR fit for one slice."""
    expiry_time: float
    sabr: SABRParams
    rmse: float
    forward_price: float
    n_quotes_total: int
    n_quotes_used: int

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