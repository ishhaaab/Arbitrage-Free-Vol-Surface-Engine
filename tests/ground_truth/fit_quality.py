"""Fit-quality harness: per-slice model-IV vs mid-IV errors.

This harness computes how well a fitted model curve reproduces a slice's
MARKET implied volatilities:

- the market IV of each quote is the implied vol of the quote's MID price
  (mid of bid/ask, falling back to ``price`` when one side is missing),
  solved with the repo's ``implied_vol``;
- ``per_strike_iv_errors`` returns ``(strike, mid_iv, model_iv, error)``
  per strike;
- ``slice_iv_rmse`` returns the per-slice root-mean-square of
  ``(model_iv - mid_iv)`` in implied-vol units.

.. note::

   This is the metric to apply to REAL market chains to judge FIT QUALITY
   (how well the model tracks the market smile).  It is NOT a correctness
   gate for the arbitrage machinery: a large per-slice RMSE means the model
   is a poor description of the market slice, not that the surface is
   arbitrageable.  The ground-truth tests use it on synthetic chains built
   from a known model to prove it can distinguish a correctly-specified
   model from a wrong one.
"""

from __future__ import annotations

from datetime import date
from math import sqrt

from arbfree_vol.models.option import ImpliedVolInput, OptionContract
from arbfree_vol.models.surface import ExpirySlice, VolSurface, get_q, get_r
from arbfree_vol.pricing.implied_vol import implied_vol


def quote_mid_price(quote) -> float:
    """Mid of bid/ask, falling back to the single ``price``."""
    if quote.bid is not None and quote.ask is not None:
        return (quote.bid + quote.ask) / 2.0
    return quote.price


def mid_iv(surface: VolSurface, sl: ExpirySlice, quote) -> float | None:
    """Implied vol of the quote's mid price (None if no root exists)."""
    iv_input = ImpliedVolInput(
        contract=OptionContract(
            symbol="_",
            option_type=quote.option_type,
            strike=quote.strike,
            expiry_date=date(2004, 1, 1),
        ),
        spot=surface.spot,
        expiry_time=sl.expiry_time,
        risk_free=get_r(surface, sl),
        div_yield=get_q(surface, sl),
        market_price=quote_mid_price(quote),
    )
    return implied_vol(iv_input)


def slice_mid_ivs(surface: VolSurface, sl: ExpirySlice) -> dict[float, float]:
    """strike -> mid IV for every quote with a solvable mid price.

    When a strike has both a call and a put the two mid IVs are averaged
    (mirroring ``slice_total_variance``'s per-strike averaging convention).
    """
    acc: dict[float, list[float]] = {}
    for q in sl.quotes:
        iv = mid_iv(surface, sl, q)
        if iv is not None:
            acc.setdefault(q.strike, []).append(iv)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def per_strike_iv_errors(
    surface: VolSurface,
    sl: ExpirySlice,
    model_iv_fn,
) -> list[tuple[float, float, float, float]]:
    """Per-strike (strike, mid_iv, model_iv, model_iv - mid_iv) tuples.

    ``model_iv_fn`` is a callable mapping an absolute strike ``K`` to the
    model's Black-Scholes implied vol for this slice (e.g. a closure over a
    fitted SVI slice's parameters and the slice's forward).
    """
    errors: list[tuple[float, float, float, float]] = []
    for strike, mid in sorted(slice_mid_ivs(surface, sl).items()):
        model = float(model_iv_fn(strike))
        errors.append((strike, mid, model, model - mid))
    return errors


def slice_iv_rmse(
    surface: VolSurface,
    sl: ExpirySlice,
    model_iv_fn,
) -> float:
    """Per-slice RMSE of (model IV - mid IV) over all solvable strikes.

    ``sqrt(mean((model_iv(K) - mid_iv(K))^2))`` in implied-vol units.
    """
    errors = per_strike_iv_errors(surface, sl, model_iv_fn)
    if not errors:
        return float("nan")
    return sqrt(sum(e[3] * e[3] for e in errors) / len(errors))
