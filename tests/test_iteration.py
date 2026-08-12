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


def test_iterative_max_iters_zero_returns_empty() -> None:
    """``max_iters=0`` runs no iterations and returns an empty list
    (documented contract in ``iterative_repair``)."""
    surface = _clean_surface(n_strikes=7)

    reports = iterative_repair(surface, max_iters=0)

    assert reports == []


def test_iterative_repair_second_pass_adds_no_new_rejections_and_surface_stable() -> None:
    """A converged run is a fixpoint whose honest contract is: re-running
    ``iterative_repair`` on the returned final cleaned surface adds NO
    new rejections (``rejected == ()`` and ``n_rejected == 0``) and
    reproduces the same final surface state (arb-free, identical
    ``n_violations_after``, ``n_slices_fitted`` and ``n_slices_input``).

    Rejection COUNTS are intentionally NOT asserted equal across passes:
    the second pass runs on the already-cleaned (smaller) surface, so
    ``n_total_quotes`` / ``n_violations_before`` / ``n_rejected`` are
    naturally lower on pass 2.  The stability claim is about the END
    STATE of the surface, not the input-dependent intermediate counts."""
    # A surface that forces a repair: a bad call at K=100 priced at 20
    # breaks put-call parity at that strike (the parity detector rejects
    # both quotes of the pair).
    quotes: list[Quote] = []
    for K in [80, 90, 100, 110, 120]:
        for o in [OptionType.CALL, OptionType.PUT]:
            quotes.append(Quote(strike=K, option_type=o, price=_bp(o, K)))
    quotes.append(Quote(strike=100.0, option_type=OptionType.CALL, price=20.0))
    surface = VolSurface(spot=SPOT, risk_free=R, div_yield=Q,
                         slices=[ExpirySlice(expiry_time=T, quotes=quotes)])

    reports = iterative_repair(surface, max_iters=5)
    final_surface = reports[-1].cleaned_surface
    assert final_surface is not None

    # The first pass's final report is the reference END STATE.  Its
    # surface fields (arb-free, n_violations_after, n_slices_fitted,
    # n_slices_input) define what "stable" means; n_rejected is
    # deliberately NOT part of the fixpoint claim.
    first_metrics = reports[-1].metrics
    # The fixture genuinely exercised the rejection path, so the
    # fixpoint surface is the product of a repair, not a trivial input.
    assert first_metrics.n_rejected > 0, (
        f"fixture must force a repair, got n_rejected="
        f"{first_metrics.n_rejected}"
    )

    reports2 = iterative_repair(final_surface, max_iters=5)
    final2 = reports2[-1]

    # The second pass on the already-cleaned surface adds NO new
    # rejections: the fixpoint surface is stable, so nothing further is
    # rejected.
    assert final2.rejected == (), (
        f"fixpoint run must reject no quotes, got {final2.rejected}"
    )
    assert final2.metrics.n_rejected == 0, (
        f"fixpoint run must reject nothing, got {final2.metrics.n_rejected}"
    )
    # The final surface state is stable across passes: the arb-free
    # result, the post-repair violation count, and the fitted/input
    # slice structure reproduce the first pass's final report exactly.
    assert final2.remaining_violations.is_arbitrage_free
    assert final2.metrics.n_violations_after == first_metrics.n_violations_after
    assert final2.metrics.n_slices_fitted == first_metrics.n_slices_fitted
    assert final2.metrics.n_slices_input == first_metrics.n_slices_input


def test_iterative_repair_stops_after_two_zero_rejections(monkeypatch) -> None:
    """The two-consecutive-zero-rejection stop branch is exercised: when
    two iterations in a row reject ZERO quotes but the surface is NOT yet
    arb-free (violations remain that cannot be rejected away), the loop
    stops after the second zero-rejection iteration instead of running to
    ``max_iters``.

    The repair step is monkeypatched with a canned non-arb-free report so
    the branch is deterministic (the real pipeline would converge to
    arb-free before reaching it)."""
    import arbfree_vol.repair.iteration as it_mod
    from arbfree_vol.arbitrage.report import (
        ArbitrageReport,
        ArbitrageViolation,
        ViolationType,
    )
    from arbfree_vol.repair.report import RepairReport, RepairMetrics

    surface = _clean_surface(n_strikes=7)
    cleaned = surface  # the fake repair never changes the surface

    def _fake_report() -> RepairReport:
        return RepairReport(
            rejected=(),
            fitted_slices=(),
            remaining_violations=ArbitrageReport(violations=[
                ArbitrageViolation(
                    kind=ViolationType.CALENDAR,
                    detail="fake remaining violation",
                    magnitude=0.1,
                ),
            ]),
            metrics=RepairMetrics(
                n_rejected=0, n_total_quotes=0,
                n_slices_input=0, n_slices_fitted=0,
                n_violations_before=0, n_violations_after=1,
            ),
            cleaned_surface=cleaned,
        )

    monkeypatch.setattr(it_mod, "repair", lambda _surface: _fake_report())

    reports = iterative_repair(surface, max_iters=5)

    # Two zero-rejection iterations, then the documented stop condition
    # (two consecutive n_rejected == 0) fires.
    assert len(reports) == 2, (
        f"expected the loop to stop after 2 zero-rejection iterations, "
        f"got {len(reports)} reports"
    )
    assert all(r.metrics.n_rejected == 0 for r in reports)
    assert all(not r.remaining_violations.is_arbitrage_free for r in reports)
