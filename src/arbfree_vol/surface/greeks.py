"""Portfolio-level Greeks computed from a fitted volatility surface."""

from dataclasses import dataclass
from datetime import date

import numpy as np

from arbfree_vol.models.option import (
    BlackScholesInput,
    OptionContract,
    OptionType,
)
from arbfree_vol.pricing.greeks import greeks as _compute_greeks
from arbfree_vol.surface.interpolate import FittedSurface, iv_at


# ---------------------------------------------------------------------------
# PortfolioGreeks — compute boundary type (frozen dataclass, no Pydantic)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PortfolioGreeks:
    """Aggregated Greeks for a portfolio of option positions.

    .. note::

        Theta is reported as the raw annualised Black-Scholes theta
        (dPrice / dT).  Divide by 365 to obtain theta per calendar day.
    """

    total_delta: float
    total_gamma: float
    total_vega: float
    total_theta: float
    total_rho: float


# ---------------------------------------------------------------------------
# Portfolio-level Greek aggregation
# ---------------------------------------------------------------------------
def portfolio_greeks(
    fs: FittedSurface,
    positions: list[tuple[OptionContract, float, float]],
    r: float | None = None,
    q: float | None = None,
) -> PortfolioGreeks:
    """Compute aggregate Greeks for a list of option positions.

    Each position is ``(contract, expiry_time, quantity)`` where
    *quantity* is signed (positive = long, negative = short).

    Parameters
    ----------
    fs:
        Fitted surface used to look up implied volatility.
    positions:
        List of (contract, expiry_time, quantity) tuples.
    r:
        Override risk-free rate (defaults to ``fs.risk_free``).
    q:
        Override dividend yield (defaults to ``fs.div_yield``).

    Returns
    -------
    PortfolioGreeks
    """
    rate = r if r is not None else fs.risk_free
    div = q if q is not None else fs.div_yield

    total_delta = 0.0
    total_gamma = 0.0
    total_vega = 0.0
    total_theta = 0.0
    total_rho = 0.0

    for contract, expiry_time, quantity in positions:
        sigma = iv_at(fs, contract.strike, expiry_time)

        model = BlackScholesInput(
            contract=contract,
            spot=fs.spot,
            expiry_time=expiry_time,
            risk_free=rate,
            div_yield=div,
            volatility=sigma,
        )

        g = _compute_greeks(model)

        total_delta += quantity * g.delta
        total_gamma += quantity * g.gamma
        total_vega += quantity * g.vega
        total_theta += quantity * g.theta
        total_rho += quantity * g.rho

    return PortfolioGreeks(
        total_delta=total_delta,
        total_gamma=total_gamma,
        total_vega=total_vega,
        total_theta=total_theta,
        total_rho=total_rho,
    )


# ---------------------------------------------------------------------------
# Bucketed Greek grids
# ---------------------------------------------------------------------------
_DUMMY_DATE = date(2030, 6, 15)
"""Dummy reference date used for ``OptionContract`` objects in bucketed
grids — the actual expiry time is passed as a float to
``BlackScholesInput`` so the date field is irrelevant."""


def bucketed_greeks(
    fs: FittedSurface,
    strikes: list[float],
    expiries: list[float],
    option_type: OptionType,
    r: float | None = None,
    q: float | None = None,
) -> dict[str, np.ndarray]:
    """Compute a 2-D grid of individual Greeks for a given option type.

    Returns 5 arrays of shape ``(len(strikes), len(expiries))`` keyed by
    ``"delta"``, ``"gamma"``, ``"vega"``, ``"theta"``, ``"rho"``.

    Parameters
    ----------
    fs:
        Fitted surface.
    strikes:
        Strike grid (rows).
    expiries:
        Expiry grid (columns), in years.
    option_type:
        Option type for every cell (``OptionType.CALL`` or ``OptionType.PUT``).
    r:
        Override risk-free rate.
    q:
        Override dividend yield.

    Returns
    -------
    dict[str, np.ndarray]
        5 arrays of shape ``(n_strikes, n_expiries)``.
    """
    n_strikes = len(strikes)
    n_expiries = len(expiries)

    delta_grid = np.zeros((n_strikes, n_expiries))
    gamma_grid = np.zeros((n_strikes, n_expiries))
    vega_grid = np.zeros((n_strikes, n_expiries))
    theta_grid = np.zeros((n_strikes, n_expiries))
    rho_grid = np.zeros((n_strikes, n_expiries))

    rate = r if r is not None else fs.risk_free
    div = q if q is not None else fs.div_yield

    for i, K in enumerate(strikes):
        for j, T in enumerate(expiries):
            contract = OptionContract(
                symbol="BUCKET",
                option_type=option_type,
                strike=K,
                expiry_date=_DUMMY_DATE,
            )
            sigma = iv_at(fs, K, T)

            model = BlackScholesInput(
                contract=contract,
                spot=fs.spot,
                expiry_time=T,
                risk_free=rate,
                div_yield=div,
                volatility=sigma,
            )

            g = _compute_greeks(model)

            delta_grid[i, j] = g.delta
            gamma_grid[i, j] = g.gamma
            vega_grid[i, j] = g.vega
            theta_grid[i, j] = g.theta
            rho_grid[i, j] = g.rho

    return {
        "delta": delta_grid,
        "gamma": gamma_grid,
        "vega": vega_grid,
        "theta": theta_grid,
        "rho": rho_grid,
    }
