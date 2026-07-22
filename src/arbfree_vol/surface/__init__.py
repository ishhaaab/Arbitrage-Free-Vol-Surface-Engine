"""Interpolation, Greeks, and scenario risk for fitted volatility surfaces."""

from arbfree_vol.surface.interpolate import (
    FittedSurface,
    build_fitted_surface,
    total_variance_at,
    iv_at,
)

from arbfree_vol.surface.greeks import (
    PortfolioGreeks,
    portfolio_greeks,
    bucketed_greeks,
)

from arbfree_vol.surface.risk import (
    ScenarioResult,
    portfolio_pnl,
    spot_bump_analysis,
    vol_bump_analysis,
    parallel_vega_pnl,
)

__all__ = [
    "FittedSurface",
    "build_fitted_surface",
    "total_variance_at",
    "iv_at",
    "PortfolioGreeks",
    "portfolio_greeks",
    "bucketed_greeks",
    "ScenarioResult",
    "portfolio_pnl",
    "spot_bump_analysis",
    "vol_bump_analysis",
    "parallel_vega_pnl",
]
