"""Mispricing signal detection — compare market IV to fitted surface IV."""

from __future__ import annotations

from datetime import date, timedelta
from math import isclose

from arbfree_vol.models.option import OptionContract, ImpliedVolInput, OptionType
from arbfree_vol.models.surface import VolSurface, get_r, get_q
from arbfree_vol.pricing.implied_vol import implied_vol
from arbfree_vol.surface.interpolate import FittedSurface, iv_at

from arbfree_vol.backtest.types import MispricingSignal

# ---------------------------------------------------------------------------
# Module-level named constants (no-hardcoding rule)
# ---------------------------------------------------------------------------
_MIN_SIGNAL_T_YEARS_DEFAULT: float = 14.0 / 365.0
"""Near-expiry options (T < 14 days) are excluded because their smiles can be
unstable and the IV solver may not converge reliably."""

_EXPIRY_MATCH_TOL: float = 1e-6
"""Tolerance for matching a surface slice expiry_time to a fitted slice."""

_DUMMY_SYMBOL: str = "BT"
"""Symbol used for ``OptionContract`` instances created inside the detector."""


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
def _is_fitted(
    expiry_time: float, fitted_expiries: list[float]
) -> bool:
    """Check if *expiry_time* matches any fitted-slice expiry (within tol)."""
    return any(
        isclose(expiry_time, fe, abs_tol=_EXPIRY_MATCH_TOL)
        for fe in fitted_expiries
    )


def detect_mispricing(
    surface: VolSurface,
    fs: FittedSurface,
    threshold: float = 0.01,
    min_signal_T_years: float = _MIN_SIGNAL_T_YEARS_DEFAULT,
    snapshot_date: date | None = None,
) -> list[MispricingSignal]:
    """Detect quotes whose market IV differs from the fitted model IV.

    Algorithm
    ---------
    1. Build a set of fitted expiry times from ``fs.fitted_slices``.
    2. For each ``ExpirySlice`` in *surface*, only process it if its
       ``expiry_time`` matches a fitted slice (within 1e-6).
    3. For every kept quote in matching slices:
       - compute market IV via ``implied_vol`` (skip if unsolvable).
       - compute model IV via ``iv_at`` (skip on ``ValueError``).
       - ``mispricing = market_iv - model_iv``.
       - skip if ``|mispricing| <= threshold``.
       - ``side = +1`` (underpriced → long) or ``-1`` (overpriced → short).
    4. Return list of signals (any order).

    Parameters
    ----------
    surface:
        Raw (or cleaned) volatility surface with market quotes.
    fs:
        Fitted arbitrage-free surface.
    threshold:
        Minimum absolute mispricing (in vol points) to generate a signal.
    min_signal_T_years:
        Minimum time-to-expiry (years) for a quote to be considered.
    snapshot_date:
        Calendar date of the snapshot.  Used to derive ``expiry_date``.
        Defaults to ``date.today()``.

    Returns
    -------
    list[MispricingSignal]
        Detected signals (empty list if none).
    """
    snapshot_date = snapshot_date or date.today()

    fitted_expiries: list[float] = [
        sl.expiry_time for sl in fs.fitted_slices
    ]

    signals: list[MispricingSignal] = []

    for sl in surface.slices:
        # Only process slices that were successfully fitted
        if not _is_fitted(sl.expiry_time, fitted_expiries):
            continue

        if sl.expiry_time < min_signal_T_years:
            continue

        r = get_r(surface, sl)
        q = get_q(surface, sl)

        expiry_date = snapshot_date + timedelta(
            days=round(sl.expiry_time * 365)
        )

        for quote in sl.quotes:
            # --- compute market IV ---
            contract = OptionContract(
                symbol=_DUMMY_SYMBOL,
                option_type=quote.option_type,
                strike=quote.strike,
                expiry_date=expiry_date,
            )
            iv_input = ImpliedVolInput(
                contract=contract,
                spot=surface.spot,
                expiry_time=sl.expiry_time,
                risk_free=r,
                div_yield=q,
                market_price=quote.price,
            )
            market_iv = implied_vol(iv_input)
            if market_iv is None:
                continue  # IV solver could not find a root

            # --- compute model IV ---
            try:
                model_iv = iv_at(fs, quote.strike, sl.expiry_time)
            except ValueError:
                continue  # defensive — shouldn't trigger for fitted slices

            mispricing = market_iv - model_iv

            if abs(mispricing) <= threshold:
                continue

            side = +1 if mispricing < 0 else -1  # underpriced → long

            signals.append(
                MispricingSignal(
                    strike=quote.strike,
                    expiry_time=sl.expiry_time,
                    expiry_date=expiry_date,
                    option_type=quote.option_type,
                    market_iv=market_iv,
                    model_iv=model_iv,
                    mispricing=mispricing,
                    entry_price=quote.price,
                    side=side,
                )
            )

    return signals
