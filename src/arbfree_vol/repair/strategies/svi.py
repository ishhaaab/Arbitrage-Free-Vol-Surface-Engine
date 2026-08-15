import logging
from math import log, sqrt
from statistics import mean

from arbfree_vol.models.fitted import FittedSlice
from arbfree_vol.models.surface import VolSurface, ExpirySlice
from arbfree_vol.svi.calibration import calibrate_constrained
from arbfree_vol.svi.model import svi_total_variance, SVIParams
from arbfree_vol.variance import slice_total_variance

from arbfree_vol.repair.strategies._common import (
    _PathFitResult,
    _PrepStatus,
    _SlicePrep,
    _prepare_slice,
    RepairStrategy,
)

_logger = logging.getLogger(__name__)


def _fit_slice(sl: ExpirySlice,
               forward_price: float,
               surface: VolSurface,
               prev_slice: SVIParams | None = None) -> FittedSlice | None:

    """Fit SVI to one cleaned slice using the estimated forward price.

    Returns None if fewer than 5 (k, w) points are available (a SKIP,
    mirroring the eSSVI path — tiny slices are neither fitted nor
    recorded as failed).  A calibration failure (``RuntimeError`` from
    ``calibrate_constrained``) PROPAGATES to the caller, which records
    the slice in ``failed_slices`` — a slice must not vanish with zero
    record.
    """
    # total variance uses the surface r/q for IV solving (independent of forward)
    strike_w= slice_total_variance(surface, sl)
    if len(strike_w) < 5:
        _logger.warning(
            "slice T=%.4f has %d (k,w) points after IV solving — need >= 5; skipping",
            sl.expiry_time, len(strike_w),
        )
        return None

    # build (k, w) points using the estimated forward, not surface r/q
    points= [
        (log(strike / forward_price), w)
        for strike, w in strike_w.items()
    ]
    points.sort()

    params= calibrate_constrained(points, prev_slice=prev_slice)

    # RMSE in w-space
    errors= [
        (svi_total_variance(k, params.a, params.b, params.rho, params.m, params.sigma) - w) ** 2
        for k, w in points
    ]
    rmse= sqrt(mean(errors))

    return FittedSlice(
        expiry_time=sl.expiry_time,
        params=params,
        rmse=rmse,
        forward_price=forward_price,
        n_quotes_total=len(sl.quotes),
        n_quotes_used=len(points),
        data_points=tuple(points),
    )


class SVIStrategy:
    """Raw-SVI path: constrained per-slice calibration.

    Fits each cleaned slice with ``_fit_slice`` under the constrained
    SVI calibration, carrying the previous slice's params as the
    calendar-penalty prior (``prev_slice``).  A ``RuntimeError`` from
    ``calibrate_constrained`` records the slice in ``failed_slices``
    rather than silently dropping it.
    """
    name = "SVI"

    def fit(self, cleaned_surface: VolSurface, fwd_curve: dict[float, float]) -> _PathFitResult:
        fitted: list[FittedSlice] = []
        failed_slices: list[float] = []
        sorted_slices = sorted(cleaned_surface.slices, key=lambda sl: sl.expiry_time)

        prev_svi: SVIParams | None = None
        for sl in sorted_slices:
            prep = _prepare_slice(sl, cleaned_surface, fwd_curve, self.name)
            if prep.status is _PrepStatus.NO_FORWARD:
                failed_slices.append(prep.expiry_time)
                continue
            if prep.status is _PrepStatus.TOO_FEW:
                continue
            assert prep.forward is not None
            try:
                fs= _fit_slice(sl, prep.forward, cleaned_surface, prev_slice=prev_svi)
            except RuntimeError as exc:
                # Honest bookkeeping: a slice whose calibration fails
                # entirely is recorded in failed_slices (mirroring the
                # eSSVI path), not silently dropped from the report.
                _logger.warning(
                    "SVI constrained calibration failed for slice T=%.4f: "
                    "%s; slice recorded as failed",
                    sl.expiry_time, exc,
                )
                failed_slices.append(sl.expiry_time)
                continue
            if fs is not None:
                fitted.append(fs)
                prev_svi = fs.params

        return _PathFitResult(fitted=fitted, failed_slices=failed_slices)
