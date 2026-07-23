"""Delta-hedged P&L realisation for a single backtest trade.

The hedge uses the **frozen-vol convention**: the entry market IV is held
constant for every daily rebalancing.  This is the standard simplification
that avoids path-dependence in the hedge vol assumption.
"""

from __future__ import annotations

from datetime import date

from arbfree_vol.models.option import (
    BlackScholesInput,
    OptionContract,
    OptionType,
)
from arbfree_vol.pricing.greeks import greeks as compute_greeks

from arbfree_vol.backtest.types import OptionType, Trade, TradePnL

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_HEDGE_DUMMY_SYMBOL: str = "HEDGE"
"""Dummy symbol used for ``OptionContract`` instances in hedge delta
computations — the actual expiry time is passed as a separate float."""


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _intrinsic(option_type: OptionType, K: float, S_T: float) -> float:
    """Payoff of a European option at expiry."""
    if option_type == OptionType.CALL:
        return max(S_T - K, 0.0)
    return max(K - S_T, 0.0)


# ---------------------------------------------------------------------------
# P&L realisation
# ---------------------------------------------------------------------------
def realize_trade_pnl(
    trade: Trade, price_path: dict[date, float]
) -> TradePnL:
    """Realise the delta-hedged P&L for a single trade held to expiry.

    Parameters
    ----------
    trade:
        The trade to realise.  The hedge vol is ``trade.frozen_vol``
        (the entry market IV), held constant for all daily rebalancings.
    price_path:
        Daily closing prices keyed by date.  Must include
        ``trade.entry_date``.  If ``trade.signal.expiry_date`` is not
        present, the last date *<= expiry_date* is used as the settlement
        date.

    Returns
    -------
    TradePnL

    Raises
    ------
    ValueError
        If *price_path* is empty or does not contain ``entry_date``.
    """
    entry_date = trade.entry_date
    expiry_date = trade.signal.expiry_date
    K = trade.signal.strike
    opt_type = trade.signal.option_type
    qty = trade.quantity
    sigma = trade.frozen_vol
    r = trade.risk_free
    q = trade.div_yield

    # ---- filter and validate price path ----
    trading_dates = sorted(
        d for d in price_path if entry_date <= d <= expiry_date
    )
    if not trading_dates or trading_dates[0] != entry_date:
        raise ValueError(
            f"price_path must include entry_date={entry_date}; "
            f"got dates {list(price_path)}"
        )

    # settlement date = last available date <= expiry_date
    settlement_date = trading_dates[-1]
    expiry_spot = price_path[settlement_date]

    # ---- option P&L at expiry ----
    option_pnl = qty * (
        _intrinsic(opt_type, K, expiry_spot) - trade.signal.entry_price
    )

    # ---- cumulative delta-hedge P&L ----
    # frozen-vol convention: hedge vol = entry market IV, held constant
    hedge_pnl = 0.0
    # Build a dummy contract — the actual expiry_time is passed separately
    dummy_contract = OptionContract(
        symbol=_HEDGE_DUMMY_SYMBOL,
        option_type=opt_type,
        strike=K,
        expiry_date=expiry_date,
    )

    for i in range(len(trading_dates) - 1):
        d_prev = trading_dates[i]
        d_curr = trading_dates[i + 1]
        S_prev = price_path[d_prev]
        S_curr = price_path[d_curr]

        # Remaining time-to-expiry at the start of the day (in years)
        t_prev = (expiry_date - d_prev).days / 365.0
        if t_prev <= 0.0:
            continue

        bs_input = BlackScholesInput(
            contract=dummy_contract,
            spot=S_prev,
            expiry_time=t_prev,
            risk_free=r,
            div_yield=q,
            volatility=sigma,
        )
        g = compute_greeks(bs_input)
        delta_prev = g.delta

        # Hedge contribution: (-qty * delta) * ΔS
        hedge_pnl += (-qty * delta_prev) * (S_curr - S_prev)

    realized_pnl = option_pnl + hedge_pnl
    hold_days = (expiry_date - entry_date).days
    hit = realized_pnl > 0.0

    return TradePnL(
        trade=trade,
        realized_pnl=realized_pnl,
        option_pnl=option_pnl,
        hedge_pnl=hedge_pnl,
        expiry_spot=expiry_spot,
        hold_days=hold_days,
        hit=hit,
    )
