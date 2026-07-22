"""Scenario risk and P&L analytics for a fitted volatility surface."""

from dataclasses import dataclass

from arbfree_vol.models.option import OptionContract
from arbfree_vol.pricing.black_scholes import price_floats
from arbfree_vol.surface.greeks import PortfolioGreeks, portfolio_greeks
from arbfree_vol.surface.interpolate import FittedSurface, iv_at


# ---------------------------------------------------------------------------
# ScenarioResult — compute boundary type (frozen dataclass, no Pydantic)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ScenarioResult:
    """Result of a single scenario shift.

    Parameters
    ----------
    spot_bump:
        Relative spot shift applied (e.g. 0.01 = +1%).  For vol-bump
        scenarios this field is set to 0.0.
    portfolio_value_before:
        Portfolio value under baseline assumptions.
    portfolio_value_after:
        Portfolio value after the scenario shift.
    pnl:
        ``value_after - value_before``.
    delta_pnl:
        Linear delta approximation ``total_delta × (Δspot)``.
    gamma_pnl:
        Second-order gamma approximation
        ``0.5 × total_gamma × (Δspot)²``.
    """

    spot_bump: float
    portfolio_value_before: float
    portfolio_value_after: float
    pnl: float
    delta_pnl: float
    gamma_pnl: float


# ---------------------------------------------------------------------------
# Pricing helper
# ---------------------------------------------------------------------------
def _price_position(
    fs: FittedSurface,
    contract: OptionContract,
    expiry_time: float,
    quantity: float,
    spot: float,
    vol_shift: float = 0.0,
) -> float:
    """Price a single option position using the fitted surface.

    Returns ``quantity × price`` (signed).  If *vol_shift* is non-zero
    it is added to the base implied volatility *before* pricing.
    """
    base_iv = iv_at(fs, contract.strike, expiry_time)
    sigma = base_iv + vol_shift

    is_call = contract.option_type.name == "CALL"
    price = price_floats(
        S=spot,
        K=contract.strike,
        T=expiry_time,
        r=fs.risk_free,
        q=fs.div_yield,
        sigma=sigma,
        is_call=is_call,
    )
    return quantity * price


def portfolio_pnl(
    fs: FittedSurface,
    positions: list[tuple[OptionContract, float, float]],
    spot_override: float | None = None,
) -> float:
    """Compute the total P&L (portfolio value) for a set of positions.

    Each position is ``(contract, expiry_time, quantity)``.

    Parameters
    ----------
    fs:
        Fitted surface.
    positions:
        List of positions.
    spot_override:
        If provided, use this spot price instead of ``fs.spot``.

    Returns
    -------
    float
        Sum of ``quantity × option_price`` over all positions.
    """
    spot = spot_override if spot_override is not None else fs.spot
    total = 0.0
    for contract, expiry_time, quantity in positions:
        total += _price_position(fs, contract, expiry_time, quantity, spot=spot)
    return total


# ---------------------------------------------------------------------------
# Scenario analyses
# ---------------------------------------------------------------------------
def spot_bump_analysis(
    fs: FittedSurface,
    positions: list[tuple[OptionContract, float, float]],
    bumps: list[float],
) -> list[ScenarioResult]:
    """Evaluate portfolio P&L under a series of relative spot shifts.

    Parameters
    ----------
    fs:
        Fitted surface.
    positions:
        List of positions.
    bumps:
        Relative spot shifts (e.g. ``[-0.01, 0.0, 0.01]``).

    Returns
    -------
    list[ScenarioResult]
        One result per bump.
    """
    base_value = portfolio_pnl(fs, positions)
    base_greeks = portfolio_greeks(fs, positions)

    results: list[ScenarioResult] = []
    for b in bumps:
        new_spot = fs.spot * (1.0 + b)
        ds = new_spot - fs.spot
        value = portfolio_pnl(fs, positions, spot_override=new_spot)
        pnl = value - base_value
        delta_pnl = base_greeks.total_delta * ds
        gamma_pnl = 0.5 * base_greeks.total_gamma * ds * ds

        results.append(
            ScenarioResult(
                spot_bump=b,
                portfolio_value_before=base_value,
                portfolio_value_after=value,
                pnl=pnl,
                delta_pnl=delta_pnl,
                gamma_pnl=gamma_pnl,
            )
        )

    return results


def vol_bump_analysis(
    fs: FittedSurface,
    positions: list[tuple[OptionContract, float, float]],
    vol_shifts: list[float],
) -> list[ScenarioResult]:
    """Evaluate portfolio P&L under parallel shifts to implied volatility.

    Each shift is added directly to every option's implied vol (in
    volatility-space, not variance-space).

    Parameters
    ----------
    fs:
        Fitted surface.
    positions:
        List of positions.
    vol_shifts:
        Volatility shifts in annualised terms (e.g. ``[0.0, 0.01, -0.01]``
        for +1/-1 percentage-point shifts).

    Returns
    -------
    list[ScenarioResult]
        One result per vol shift.  ``spot_bump`` is set to 0.0 for all
        entries (the shock is in vol, not spot).  ``delta_pnl`` and
        ``gamma_pnl`` are set to 0.0 since this is not a spot shock.
    """
    base_value = portfolio_pnl(fs, positions)

    results: list[ScenarioResult] = []
    for shift in vol_shifts:
        total = 0.0
        for contract, expiry_time, quantity in positions:
            total += _price_position(
                fs, contract, expiry_time, quantity,
                spot=fs.spot, vol_shift=shift,
            )
        pnl = total - base_value

        results.append(
            ScenarioResult(
                spot_bump=0.0,
                portfolio_value_before=base_value,
                portfolio_value_after=total,
                pnl=pnl,
                delta_pnl=0.0,
                gamma_pnl=0.0,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Parallel vega P&L approximation
# ---------------------------------------------------------------------------
def parallel_vega_pnl(
    fs: FittedSurface,
    positions: list[tuple[OptionContract, float, float]],
    vega_shift: float,
) -> float:
    """Approximate P&L from a parallel vol shift using aggregate vega.

    The result is ``portfolio_vega × vega_shift``, where
    *vega_shift* is the parallel change in annualised implied volatility.

    .. note::

        This is a first-order approximation — it ignores gamma-vol and
        cross-gamma effects.  Use ``vol_bump_analysis`` for a full
        re-pricing.
    """
    pg: PortfolioGreeks = portfolio_greeks(fs, positions)
    return pg.total_vega * vega_shift
