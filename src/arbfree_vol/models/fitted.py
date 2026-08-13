"""Fitted-surface types shared by repair, surface, and pricing.

These frozen dataclasses describe the OUTPUT of smile-model calibration:
the raw-SVI parameters per slice (the common currency every model maps
to), plus the native eSSVI / SABR parameters.  They live in ``models`` —
not ``repair`` — so the low-level ``surface`` and ``pricing`` layers can
depend on them without depending on the ``repair`` orchestrator.
"""

from dataclasses import dataclass

from arbfree_vol.sabr.model import SABRParams
from arbfree_vol.ssvi.model import SSVIParams, eSSVISurfaceParams
from arbfree_vol.svi.model import SVIParams


@dataclass(frozen=True, slots=True)
class FittedSlice:
    expiry_time: float
    params: SVIParams
    rmse: float
    forward_price: float
    n_quotes_total: int
    n_quotes_used: int
    data_points: tuple[tuple[float, float], ...] | None = None


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
class FittedSurface:
    """Stripped-down fitted vol surface for analytics.

    All three smile-model code paths (SVI / eSSVI / SABR) funnel their
    fitted parameters through raw SVI ``FittedSlice`` objects, so
    ``FittedSurface`` works uniformly regardless of which model was used
    during repair.
    """
    spot: float
    risk_free: float
    div_yield: float
    forward_curve: tuple[tuple[float, float], ...]
    fitted_slices: tuple[FittedSlice, ...]
