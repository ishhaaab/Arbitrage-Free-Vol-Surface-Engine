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

    Branch contract:

    - Newton runs at most ``_NEWTON_ITERS`` iterations from the
      ``sqrt(2*pi/T) * price / S`` starting point and returns as soon as
      ``|price(sigma) - target| < _NEWTON_TOL``.  It aborts (falling
      through to Brent) when vega is non-positive, or when the iterate
      leaves ``(0, high * 1.5]``.
    - The ``low``/``high`` pair is the Brent bracket: if the price admits
      no root inside ``[low, high]`` (``f(low) * f(high) > 0``), ``None``
      is returned.  The Newton fast-path is NOT bracketed by ``low``/
      ``high`` — it may return a converged value above ``high`` (or below
      it) without consulting the bracket; ``high * 1.5`` only guards
      against divergence.
    - ``None`` is also returned for prices that are unreachable at any
      sigma in the bracket (e.g. a call priced at or beyond the
      discounted spot).
    - For prices where the BS price is flat in sigma beyond double
      precision (deep ITM/OTM quotes whose vega contribution vanishes),
      the returned sigma is not unique; the solver returns whichever
      value satisfies its price tolerance, which may be a bracket
      endpoint.
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
            return sigma
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
    return cast(float, brentq(f, low, high))
