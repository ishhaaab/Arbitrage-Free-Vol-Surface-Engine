import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from math import log
from typing import Protocol

from arbfree_vol.models.fitted import FittedSlice, FittedSSVISlice, FittedSABRSlice
from arbfree_vol.models.surface import VolSurface, ExpirySlice
from arbfree_vol.variance import slice_total_variance

_logger = logging.getLogger(__name__)


class _PrepStatus(Enum):
    OK = auto()
    NO_FORWARD = auto()
    TOO_FEW = auto()


@dataclass(frozen=True)
class _SlicePrep:
    status: _PrepStatus
    expiry_time: float
    forward: float | None = None
    points: tuple[tuple[float, float], ...] = ()


@dataclass
class _PathFitResult:
    """Accumulated per-path fit outcome, replacing heterogeneous tuples.

    Each ``_repair_*`` helper returns one of these with only the fields
    relevant to its model populated; ``repair()`` reads them by name.
    Adding a new bookkeeping dimension is now a one-file change instead
    of a signature + unpacking + report ripple.
    """
    fitted: list[FittedSlice]
    failed_slices: list[float]
    fitted_ssvi: list[FittedSSVISlice] = field(default_factory=list)
    fitted_sabr: list[FittedSABRSlice] = field(default_factory=list)
    fallback_slices: list[float] = field(default_factory=list)
    sabr_mapping_failed: list[float] = field(default_factory=list)
    repair_infeasible: bool = False


def _prepare_slice(
    sl: ExpirySlice,
    surface: VolSurface,
    fwd_curve: dict[float, float],
    path: str,
) -> _SlicePrep:
    """Shared per-slice prep for all three repair paths.

    Applies the identical bookkeeping semantics each path used to
    implement inline: a slice with no forward estimate is a FAILURE
    (recorded by the caller in its failed list), a slice with fewer
    than 5 (k, w) points is a SKIP (neither fitted nor recorded),
    and the forward check wins over the point-count check.  ``path``
    is the model name used in the no-forward warning (``"SVI"``,
    ``"eSSVI"``, ``"SABR"``) so the per-path log text is preserved.
    """
    F = fwd_curve.get(sl.expiry_time)
    if F is None:
        _logger.warning(
            "%s path: no forward estimate for slice T=%.4f; "
            "slice recorded as failed",
            path, sl.expiry_time,
        )
        return _SlicePrep(_PrepStatus.NO_FORWARD, sl.expiry_time)

    strike_w = slice_total_variance(surface, sl)
    if len(strike_w) < 5:
        _logger.warning(
            "slice T=%.4f has %d (k,w) points — need >= 5; skipping",
            sl.expiry_time, len(strike_w),
        )
        return _SlicePrep(_PrepStatus.TOO_FEW, sl.expiry_time)

    pts = sorted(
        (log(strike / F), w)
        for strike, w in strike_w.items()
    )
    return _SlicePrep(_PrepStatus.OK, sl.expiry_time, forward=F, points=tuple(pts))


class RepairStrategy(Protocol):
    """Fit contract implemented by every model repair strategy.

    ``name`` is the per-path log prefix (``"SVI"``, ``"eSSVI"``,
    ``"SABR"``); its exact value is load-bearing — the no-forward
    warning text is asserted by tests.  ``fit`` consumes the cleaned
    surface and forward curve and returns the accumulated outcome.
    """
    name: str
    def fit(self, cleaned_surface: VolSurface, fwd_curve: dict[float, float]) -> _PathFitResult: ...
