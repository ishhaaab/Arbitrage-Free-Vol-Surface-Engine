"""Tests for the iterative repair loop."""

from datetime import date

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.repair.iteration import iterative_repair
from arbfree_vol.repair.engine import repair


SPOT = 100.0
R = 0.05
Q = 0.0
T = 1.0
_DUMMY = date(2030, 1, 1)


def _bp(otype: OptionType, K: float, sigma: float = 0.2, tt: float = T) -> float:
    from arbfree_vol.models.option import OptionContract, BlackScholesInput
    from arbfree_vol.pricing.black_scholes import price
    c = OptionContract(symbol="X", option_type=otype, strike=K, expiry_date=_DUMMY)
    m = BlackScholesInput(contract=c, spot=SPOT, expiry_time=tt,
                          risk_free=R, div_yield=Q, volatility=sigma)
    return price(m)


def _clean_surface(n_strikes: int = 7) -> VolSurface:
    strikes = [SPOT * (1 + 0.1 * (i - n_strikes // 2)) for i in range(n_strikes)]
    quotes: list[Quote] = []
    for K in strikes:
        for o in [OptionType.CALL, OptionType.PUT]:
            quotes.append(Quote(strike=K, option_type=o, price=_bp(o, K)))
    return VolSurface(spot=SPOT, risk_free=R, div_yield=Q,
                      slices=[ExpirySlice(expiry_time=T, quotes=quotes)])


def test_iterative_clean_surface_converges_in_one() -> None:
    surface = _clean_surface(n_strikes=7)

    reports = iterative_repair(surface, max_iters=5)

    assert len(reports) == 1
    assert reports[-1].remaining_violations.is_arbitrage_free


def test_iterative_bad_quote_gets_rejected() -> None:
    # A clean slice + a bad call at higher strike.
    quotes: list[Quote] = []
    for K in [80, 90, 100, 110, 120]:
        for o in [OptionType.CALL, OptionType.PUT]:
            quotes.append(Quote(strike=K, option_type=o, price=_bp(o, K)))
    # bad quote: strike=100 call priced at 20 (should be ~10.45)
    quotes.append(Quote(strike=100.0, option_type=OptionType.CALL, price=20.0))

    surface = VolSurface(spot=SPOT, risk_free=R, div_yield=Q,
                         slices=[ExpirySlice(expiry_time=T, quotes=quotes)])

    reports = iterative_repair(surface, max_iters=5)

    assert len(reports) >= 1
    # The bad quote should appear in the rejected list of the first iteration
    assert any(
        r.strike == 100.0 and r.option_type == OptionType.CALL
        for r in reports[0].rejected
    )


def test_iterative_max_iters_respected() -> None:
    # A surface that has no SVI-fittable quotes (only 2 quotes).
    surface = VolSurface(spot=SPOT, risk_free=R, div_yield=Q,
                         slices=[ExpirySlice(
                             expiry_time=T,
                             quotes=[
                                 Quote(strike=100.0, option_type=OptionType.CALL, price=10.0),
                                 Quote(strike=110.0, option_type=OptionType.CALL, price=5.0),
                             ],
                         )])

    reports = iterative_repair(surface, max_iters=3)

    # Should iterate without crashing, even if no quotes can be fitted.
    assert len(reports) <= 3
    assert len(reports[-1].fitted_slices) == 0
