"""Structured output from raw SVI and constrained SSVI calibration."""

from dataclasses import dataclass

from arbfree_vol.ingestion.cleaning import RejectionRecord
from arbfree_vol.models.fitted import FittedSlice, FittedSSVISlice
from arbfree_vol.models.surface import VolSurface


@dataclass(frozen=True, slots=True)
class NumericalCertificate:
    """Finite-grid static-arbitrage check over a declared k domain."""

    k_min: float
    k_max: float
    grid_size: int
    tolerance: float
    min_total_variance: float
    min_butterfly_density: float
    min_calendar_margin: float | None
    certified: bool


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Results and failure state for one reproducible calibration run."""

    dataset_source: str
    dataset_timestamp: str
    dependency_versions: tuple[tuple[str, str], ...]
    input_quotes: int
    accepted_quotes: int
    rejected: tuple[RejectionRecord, ...]
    forwards: tuple[tuple[float, float], ...]
    raw_svi_slices: tuple[FittedSlice, ...]
    raw_svi_failed_slices: tuple[float, ...]
    fitted_slices: tuple[FittedSlice, ...]
    fitted_ssvi_slices: tuple[FittedSSVISlice, ...]
    fallback_slices: tuple[float, ...]
    failed_slices: tuple[float, ...]
    certificate: NumericalCertificate
    runtime_seconds: float
    cleaned_surface: VolSurface

    @property
    def raw_svi_rmse(self) -> float | None:
        if not self.raw_svi_slices:
            return None
        return sum(item.rmse for item in self.raw_svi_slices) / len(self.raw_svi_slices)

    @property
    def constrained_ssvi_rmse(self) -> float | None:
        if not self.fitted_ssvi_slices:
            return None
        return sum(item.rmse for item in self.fitted_ssvi_slices) / len(
            self.fitted_ssvi_slices
        )

    @property
    def rejection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.rejected:
            counts[item.rule.value] = counts.get(item.rule.value, 0) + 1
        return counts
