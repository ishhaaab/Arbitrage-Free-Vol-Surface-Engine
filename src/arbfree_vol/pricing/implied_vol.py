"""Implied volatility solver with Newton fast-path and Brent fallback."""

from typing import cast
from math import sqrt, pi

from scipy.optimize import brentq

from arbfree_vol.models.option import ImpliedVolInput, OptionType
from arbfree_vol.pricing.black_scholes import price_floats
from arbfree_vol.pricing._core import vega_floats


_NEWTON_ITERS = 5
_NEWTON_TOL = 1e-8


def implied_vol(model: ImpliedVolInput, low: float = 1e-6, high: float = 5.0,) -> float | None:
    """Solve implied volatility from a market price.

    Uses Newton-Raphson (with analytic vega) as a fast-path, then falls
    back to Brent if Newton fails to converge.

    Return contract:

    - Every returned sigma lies inside ``[low, high]``.  A computed root
      that falls outside the bracket is treated exactly like "no root in
      bracket": the function returns ``None`` — it never clamps.
    - ``None`` is also returned when the price admits no root inside
      ``[low, high]`` (``f(low) * f(high) > 0`` on the Brent path) or
      when the price is unreachable at any sigma in the bracket (e.g. a
      call priced at or beyond the discounted spot).
    - Deep ITM/OTM prices: the BS price can be locally flat in sigma
      beyond double precision (the vega contribution vanishes), so the
      implied volatility is NOT identifiable there — many sigmas
      reproduce the same price.  The function returns a root within
      ``[low, high]`` when one exists, even if it is non-unique; when no
      root exists within bounds (including a computed root that lands
      outside bounds), it returns ``None``.

    Branch notes:

    - Newton runs at most ``_NEWTON_ITERS`` iterations from the
      ``sqrt(2*pi/T) * price / S`` starting point and returns as soon as
      ``|price(sigma) - target| < _NEWTON_TOL`` AND ``low <= sigma <=
      high``.  It aborts (falling through to Brent) when vega is
      non-positive, or when the iterate leaves ``(0, high * 1.5]``.
    - The ``low``/``high`` pair brackets Brent; the ``brentq`` root is
      re-checked against the bounds before being returned.
    """
    S = model.spot
    K = model.contract.strike
    T = model.expiry_time
    r = model.risk_free
    q = model.div_yield
    is_call = model.contract.option_type == OptionType.CALL
    target = model.market_price

    # ---------- Newton fast-path ----------
    sigma = sqrt(2.0 * pi / T) * target / S

    for _ in range(_NEWTON_ITERS):
        p = price_floats(S, K, T, r, q, sigma, is_call)
        diff = p - target
        if abs(diff) < _NEWTON_TOL:
            if low <= sigma <= high:
                return sigma
            return None
        v = vega_floats(S, K, T, r, q, sigma)
        if v <= 0.0:
            break
        sigma -= diff / v
        if sigma <= 0.0 or sigma > high * 1.5:
            break

    # ---------- Brent fallback ----------
    def f(s: float) -> float:
        return price_floats(S, K, T, r, q, s, is_call) - target

    f_low = f(low)
    f_high = f(high)
    if f_low * f_high > 0:
        return None
    root = cast(float, brentq(f, low, high))
    if low <= root <= high:
        return root
    return None
