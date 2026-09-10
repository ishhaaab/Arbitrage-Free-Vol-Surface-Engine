"""Fitted raw SVI and sequential SSVI surface types."""

from dataclasses import dataclass

from arbfree_vol.ssvi.model import SSVIParams
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
    """Sequentially constrained SSVI fit for one expiry."""
    expiry_time: float
    ssvi: SSVIParams
    rmse: float
    forward_price: float
    n_quotes_total: int
    n_quotes_used: int


@dataclass(frozen=True, slots=True)
class FittedSurface:
    """Queryable fitted surface using raw-SVI-equivalent slices."""
    spot: float
    risk_free: float
    div_yield: float
    forward_curve: tuple[tuple[float, float], ...]
    fitted_slices: tuple[FittedSlice, ...]
