"""Shared fixtures and surface-builder helpers for the repair-engine tests.

These build ``VolSurface`` objects priced from Black-Scholes, raw SVI, or
eSSVI ground truth so the ``(k, w)`` data the engine sees matches the fitted
model's conventions exactly.  Kept here (rather than in a single test module)
so the repair test suite can be split per model path without duplicating the
builders.
"""
from datetime import date

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType


SPOT = 100.0
R = 0.05
Q = 0.0
T = 1.0
_DUMMY_DATE = date(2030, 1, 1)


def _bs_price(otype: OptionType, strike: float,
              sigma: float = 0.2, tt: float = T) -> float:
    from arbfree_vol.models.option import OptionContract, BlackScholesInput
    from arbfree_vol.pricing.black_scholes import price

    contract = OptionContract(
        symbol="X", option_type=otype, strike=strike,
        expiry_date=_DUMMY_DATE,
    )
    model = BlackScholesInput(
        contract=contract, spot=SPOT, expiry_time=tt,
        risk_free=R, div_yield=Q, volatility=sigma,
    )
    return price(model)


def _clean_surface(n_strikes: int = 7) -> VolSurface:
    """Build a surface with calls and puts across n_strikes, all priced at
    sigma=0.2 from the same model — no arb violations by construction."""
    strikes = [SPOT * (1 + 0.1 * (i - n_strikes // 2)) for i in range(n_strikes)]
    quotes: list[Quote] = []
    for K in strikes:
        quotes.append(
            Quote(strike=K, option_type=OptionType.CALL,
                  price=_bs_price(OptionType.CALL, K))
        )
        quotes.append(
            Quote(strike=K, option_type=OptionType.PUT,
                  price=_bs_price(OptionType.PUT, K))
        )

    return VolSurface(
        spot=SPOT, risk_free=R, div_yield=Q,
        slices=[ExpirySlice(expiry_time=T, quotes=quotes)],
    )


_DIP_TRUTH_ENGINE = [
    (0.25, dict(theta=0.08, rho=-0.3, psi=0.5)),
    (0.50, dict(theta=0.03, rho=-0.2, psi=0.35)),  # theta + chi dip
    (1.00, dict(theta=0.12, rho=0.1, psi=0.4)),
    (2.00, dict(theta=0.07, rho=0.2, psi=0.55)),   # theta dip again
]


def _ssvi_priced_surface(truth, n_strikes: int | None = None) -> VolSurface:
    """Price a surface from SSVI ground truth so the (k, w) data the
    engine sees matches the fitted model's conventions exactly."""
    from math import sqrt, exp
    from arbfree_vol.ssvi.model import ssvi_w

    ks = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    if n_strikes is not None:
        ks = ks[:n_strikes]
    slices: list[ExpirySlice] = []
    for T, t in truth:
        F = SPOT * exp((R - Q) * T)
        quotes: list[Quote] = []
        for k in ks:
            K = F * exp(k)
            w = ssvi_w(k, t["theta"], t["rho"], t["psi"])
            sigma = sqrt(max(w / T, 1e-12))
            quotes.append(Quote(
                strike=K, option_type=OptionType.CALL,
                price=_bs_price(OptionType.CALL, K, sigma=sigma, tt=T),
            ))
            quotes.append(Quote(
                strike=K, option_type=OptionType.PUT,
                price=_bs_price(OptionType.PUT, K, sigma=sigma, tt=T),
            ))
        slices.append(ExpirySlice(expiry_time=T, quotes=quotes))
    return VolSurface(spot=SPOT, risk_free=R, div_yield=Q, slices=slices)


# Two expiries sharing one valid raw-SVI parameter set (a=0.04, b=0.4,
# rho=-0.4, m=0.05, sigma=0.15).  The values themselves are not
# load-bearing: they just need to produce a well-formed,
# non-arbitrageable smile so the bookkeeping tests (failed-slice
# recording, few-point skip) run on valid input.  Two expiries (0.25, 1.0)
# let those tests assert per-slice accounting.
_SVI_TRUTH_ENGINE = [
    (0.25, dict(a=0.04, b=0.4, rho=-0.4, m=0.05, sigma=0.15)),
    (1.00, dict(a=0.04, b=0.4, rho=-0.4, m=0.05, sigma=0.15)),
]


def _svi_priced_surface(truth, n_strikes: int | None = None) -> VolSurface:
    """Price a surface from raw SVI ground truth so the (k, w) data the
    engine sees matches the fitted model's conventions exactly."""
    from math import sqrt, exp
    from arbfree_vol.svi.model import svi_total_variance

    ks = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    if n_strikes is not None:
        ks = ks[:n_strikes]
    slices: list[ExpirySlice] = []
    for T, t in truth:
        F = SPOT * exp((R - Q) * T)
        quotes: list[Quote] = []
        for k in ks:
            K = F * exp(k)
            w = svi_total_variance(k, t["a"], t["b"], t["rho"], t["m"], t["sigma"])
            sigma = sqrt(max(w / T, 1e-12))
            quotes.append(Quote(
                strike=K, option_type=OptionType.CALL,
                price=_bs_price(OptionType.CALL, K, sigma=sigma, tt=T),
            ))
            quotes.append(Quote(
                strike=K, option_type=OptionType.PUT,
                price=_bs_price(OptionType.PUT, K, sigma=sigma, tt=T),
            ))
        slices.append(ExpirySlice(expiry_time=T, quotes=quotes))
    return VolSurface(spot=SPOT, risk_free=R, div_yield=Q, slices=slices)


def _flat_bs_surface(expiries: list[float], n_strikes: int = 7) -> VolSurface:
    """Build a clean multi-slice surface priced at flat BS vol 0.2.

    Mirrors the surface construction used by the existing SABR repair
    tests (``test_sabr_failure_marks_failed_slices`` etc.) so the
    mapping-failure tests exercise the same pipeline path.
    """
    strikes = [SPOT * (1 + 0.1 * (i - n_strikes // 2)) for i in range(n_strikes)]

    slices: list[ExpirySlice] = []
    for T_val in expiries:
        quotes: list[Quote] = []
        for K in strikes:
            quotes.append(
                Quote(strike=K, option_type=OptionType.CALL,
                      price=_bs_price(OptionType.CALL, K, sigma=0.2, tt=T_val))
            )
            quotes.append(
                Quote(strike=K, option_type=OptionType.PUT,
                      price=_bs_price(OptionType.PUT, K, sigma=0.2, tt=T_val))
            )
        slices.append(ExpirySlice(expiry_time=T_val, quotes=quotes))

    return VolSurface(spot=SPOT, risk_free=R, div_yield=Q, slices=slices)


def _forward_curve_missing(expiry_to_drop: float):
    """Build a fake ``estimate_forward_curve`` that omits one expiry.

    Returns a function matching repair()'s call signature
    ``estimate_forward_curve(cleaned_surface)``.  Every expiry except
    ``expiry_to_drop`` maps to the theoretical forward
    ``spot * exp((r - q) * T)``; the dropped expiry is absent from the
    dict, which makes ``fwd_curve.get(T)`` return None in the repair
    loops.

    The fake is installed at the engine's own import site
    (``arbfree_vol.repair.engine.estimate_forward_curve``), so ONLY
    repair()'s step-4 call is affected — ``detect_with_forward`` keeps
    its own module-level import of the real estimator, so violation
    detection stays deterministic.
    """
    from math import exp

    def _estimate(surface):
        curve = {}
        for sl in surface.slices:
            if sl.expiry_time == expiry_to_drop:
                continue
            curve[sl.expiry_time] = surface.spot * exp((R - Q) * sl.expiry_time)
        return curve

    return _estimate


def _dip_truth_surface(n_strikes: int | None = None) -> VolSurface:
    """The canonical theta-dip surface (``_DIP_TRUTH_ENGINE``) priced for
    the engine — ``repair(_dip_truth_surface(), use_ssvi=True)``."""
    return _ssvi_priced_surface(_DIP_TRUTH_ENGINE, n_strikes=n_strikes)


def _svi_truth_surface() -> VolSurface:
    """The canonical raw-SVI truth surface (``_SVI_TRUTH_ENGINE``) priced
    for the engine."""
    return _svi_priced_surface(_SVI_TRUTH_ENGINE)
