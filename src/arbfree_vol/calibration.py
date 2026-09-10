"""End-to-end raw SVI baseline and constrained SSVI calibration."""

from importlib.metadata import PackageNotFoundError, version
from math import log
from time import perf_counter

from arbfree_vol.forward import estimate_forward_curve
from arbfree_vol.ingestion.cleaning import RejectionRecord
from arbfree_vol.models.surface import VolSurface
from arbfree_vol.repair.strategies import SSVIStrategy, SVIStrategy
from arbfree_vol.report import CalibrationReport
from arbfree_vol.verification import verify_surface


def _versions() -> tuple[tuple[str, str], ...]:
    found: list[tuple[str, str]] = []
    for package in ("numpy", "scipy", "pydantic", "matplotlib"):
        try:
            found.append((package, version(package)))
        except PackageNotFoundError:
            found.append((package, "not-installed"))
    return tuple(found)


def calibrate_surface(
    surface: VolSurface,
    *,
    rejected: list[RejectionRecord] | tuple[RejectionRecord, ...] = (),
    dataset_source: str = "unspecified",
    dataset_timestamp: str = "unspecified",
    k_min: float = -1.5,
    k_max: float = 1.5,
    grid_size: int = 241,
    tolerance: float = 1e-4,
) -> CalibrationReport:
    """Fit the baseline and primary model, then issue a numerical certificate."""
    started = perf_counter()
    working_surface = surface.model_copy(deep=True)
    forwards = estimate_forward_curve(working_surface)
    for expiry_slice in working_surface.slices:
        forward = forwards[expiry_slice.expiry_time]
        expiry_slice.div_yield = working_surface.risk_free - log(
            forward / working_surface.spot
        ) / expiry_slice.expiry_time
    raw_result = SVIStrategy().fit(working_surface, forwards)
    ssvi_result = SSVIStrategy().fit(working_surface, forwards)
    failures = tuple(sorted(set(ssvi_result.failed_slices)))
    fallbacks = tuple(sorted(set(ssvi_result.fallback_slices)))
    certificate = verify_surface(
        ssvi_result.fitted,
        k_min=k_min,
        k_max=k_max,
        grid_size=grid_size,
        tolerance=tolerance,
        has_fallbacks=bool(fallbacks),
        has_failures=bool(failures),
    )
    accepted = sum(len(item.quotes) for item in surface.slices)
    rejected_tuple = tuple(rejected)
    return CalibrationReport(
        dataset_source=dataset_source,
        dataset_timestamp=dataset_timestamp,
        dependency_versions=_versions(),
        input_quotes=accepted + len(rejected_tuple),
        accepted_quotes=accepted,
        rejected=rejected_tuple,
        forwards=tuple(sorted(forwards.items())),
        raw_svi_slices=tuple(raw_result.fitted),
        raw_svi_failed_slices=tuple(sorted(set(raw_result.failed_slices))),
        fitted_slices=tuple(ssvi_result.fitted),
        fitted_ssvi_slices=tuple(ssvi_result.fitted_ssvi),
        fallback_slices=fallbacks,
        failed_slices=failures,
        certificate=certificate,
        runtime_seconds=perf_counter() - started,
        cleaned_surface=working_surface,
    )
