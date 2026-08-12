"""Tests for static arbitrage detection."""

from datetime import date

import numpy as np
from pytest import approx

from arbfree_vol.arbitrage.quote_detect import detect, _check_wide_spread
from arbfree_vol.arbitrage.report import ArbitrageReport, ArbitrageViolation, ViolationType
from arbfree_vol.models.option import (
    BlackScholesInput,
    OptionContract,
    OptionType,
    OffendingQuote,
)
from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface
from arbfree_vol.pricing.black_scholes import price

SPOT = 100.0
RISK_FREE = 0.05
DIV_YIELD = 0.0
T = 1.0


def _bs_price(option_type: OptionType, strike: float, sigma: float = 0.2) -> float:
    contract = OptionContract(
        symbol="NVDA",
        option_type=option_type,
        strike=strike,
        expiry_date=date(2026, 11, 27),
    )
    model = BlackScholesInput(
        contract=contract,
        spot=SPOT,
        expiry_time=T,
        risk_free=RISK_FREE,
        div_yield=DIV_YIELD,
        volatility=sigma,
    )
    return price(model)


def _surface(quotes: list[Quote]) -> VolSurface:
    return VolSurface(
        spot=SPOT,
        risk_free=RISK_FREE,
        div_yield=DIV_YIELD,
        slices=[ExpirySlice(expiry_time=T, quotes=quotes)],
    )


def test_parity_consistent_surface_is_arbitrage_free() -> None:
    # Call and put generated from the same model => parity holds by construction.
    call = _bs_price(OptionType.CALL, 100.0)
    put = _bs_price(OptionType.PUT, 100.0)
    surface = _surface(
        [
            Quote(strike=100.0, option_type=OptionType.CALL, price=call),
            Quote(strike=100.0, option_type=OptionType.PUT, price=put),
        ]
    )

    report = detect(surface)

    assert report.is_arbitrage_free
    assert report.violations == []


def test_parity_violation_is_detected() -> None:
    call = _bs_price(OptionType.CALL, 100.0)
    put = _bs_price(OptionType.PUT, 100.0)
    surface = _surface(
        [
            Quote(strike=100.0, option_type=OptionType.CALL, price=call),
            # bump the put by 1.0 -> breaks C - P by exactly 1.0
            Quote(strike=100.0, option_type=OptionType.PUT, price=put + 1.0),
        ]
    )

    report = detect(surface)

    assert not report.is_arbitrage_free
    assert len(report.violations) == 1
    v = report.violations[0]
    assert v.kind == ViolationType.PARITY
    assert v.magnitude == approx(1.0, abs=1e-6)


def test_monotonicity_violation_is_detected() -> None:
    # Call prices must fall as strike rises; here the 110 call is dearer than the
    # 100 call -> a vertical-spread arbitrage.
    surface = _surface(
        [
            Quote(strike=100.0, option_type=OptionType.CALL, price=5.0),
            Quote(strike=110.0, option_type=OptionType.CALL, price=6.0),
        ]
    )

    report = detect(surface)

    assert not report.is_arbitrage_free
    kinds = [v.kind for v in report.violations]
    assert ViolationType.MONOTONICITY in kinds
    mono = next(v for v in report.violations if v.kind == ViolationType.MONOTONICITY)
    assert mono.magnitude == approx(1.0, abs=1e-6)


def test_monotonic_calls_are_arbitrage_free() -> None:
    # Properly decreasing call prices across strikes -> no monotonicity violation.
    surface = _surface(
        [
            Quote(strike=90.0, option_type=OptionType.CALL, price=_bs_price(OptionType.CALL, 90.0)),
            Quote(strike=100.0, option_type=OptionType.CALL, price=_bs_price(OptionType.CALL, 100.0)),
            Quote(strike=110.0, option_type=OptionType.CALL, price=_bs_price(OptionType.CALL, 110.0)),
        ]
    )

    report = detect(surface)

    assert report.is_arbitrage_free


def test_butterfly_violation_is_detected() -> None:
    # Strikes 90/100/110 (evenly spaced -> line = average of outer two = 6.0).
    # A middle call of 8.0 pokes 2.0 above the line -> negative density.
    surface = _surface(
        [
            Quote(strike=90.0, option_type=OptionType.CALL, price=10.0),
            Quote(strike=100.0, option_type=OptionType.CALL, price=8.0),
            Quote(strike=110.0, option_type=OptionType.CALL, price=2.0),
        ]
    )

    report = detect(surface)

    assert not report.is_arbitrage_free
    fly = next(v for v in report.violations if v.kind == ViolationType.BUTTERFLY)
    assert fly.magnitude == approx(2.0, abs=1e-6)


def test_convex_calls_are_arbitrage_free() -> None:
    # Call prices from a flat-vol model are convex in strike by construction.
    surface = _surface(
        [
            Quote(strike=90.0, option_type=OptionType.CALL, price=_bs_price(OptionType.CALL, 90.0)),
            Quote(strike=100.0, option_type=OptionType.CALL, price=_bs_price(OptionType.CALL, 100.0)),
            Quote(strike=110.0, option_type=OptionType.CALL, price=_bs_price(OptionType.CALL, 110.0)),
        ]
    )

    report = detect(surface)

    assert report.is_arbitrage_free


def _call_quote(strike: float, sigma: float, t: float) -> Quote:
    """A call quote priced at a given vol and maturity."""
    contract = OptionContract(
        symbol="NVDA",
        option_type=OptionType.CALL,
        strike=strike,
        expiry_date=date(2026, 11, 27),
    )
    model = BlackScholesInput(
        contract=contract,
        spot=SPOT,
        expiry_time=t,
        risk_free=RISK_FREE,
        div_yield=DIV_YIELD,
        volatility=sigma,
    )
    return Quote(strike=strike, option_type=OptionType.CALL, price=price(model))


def _two_expiry_surface(t1: float, sig1: float, t2: float, sig2: float) -> VolSurface:
    # Two strikes at two maturities, each priced at its own vol, so the
    # calendar check has enough points for k-space interpolation (the old
    # invalid single-strike fallback was removed).
    return VolSurface(
        spot=SPOT,
        risk_free=RISK_FREE,
        div_yield=DIV_YIELD,
        slices=[
            ExpirySlice(expiry_time=t1, quotes=[
                _call_quote(95.0, sig1, t1),
                _call_quote(100.0, sig1, t1),
            ]),
            ExpirySlice(expiry_time=t2, quotes=[
                _call_quote(95.0, sig2, t2),
                _call_quote(100.0, sig2, t2),
            ]),
        ],
    )


def test_calendar_violation_is_detected() -> None:
    # Short expiry (T=0.5, sigma=0.40 -> w=0.08) carries MORE total variance than
    # the long expiry (T=1.0, sigma=0.20 -> w=0.04) -> calendar arbitrage.
    surface = _two_expiry_surface(t1=0.5, sig1=0.40, t2=1.0, sig2=0.20)

    report = detect(surface)

    cal = next(v for v in report.violations if v.kind == ViolationType.CALENDAR)
    assert cal.magnitude == approx(0.08 - 0.04, abs=1e-4)


def test_increasing_total_variance_is_arbitrage_free() -> None:
    # Total variance rises with maturity (w=0.02 -> w=0.04) -> no calendar arb.
    surface = _two_expiry_surface(t1=0.5, sig1=0.20, t2=1.0, sig2=0.20)

    report = detect(surface)

    assert report.is_arbitrage_free


def test_unpaired_strike_is_skipped() -> None:
    # Only a call at this strike => parity cannot be checked, no violation.
    call = _bs_price(OptionType.CALL, 100.0)
    surface = _surface(
        [Quote(strike=100.0, option_type=OptionType.CALL, price=call)]
    )

    report = detect(surface)

    assert report.is_arbitrage_free


def test_calendar_check_handles_empty_total_variance_slice(monkeypatch) -> None:
    """A slice whose total-variance dict is empty must not crash the
    calendar check.

    Regression for the zip-unpack bug where ``ks_l, vs_l = zip(*lw)``
    ran before the ``len(lw) < 2`` guard, raising
    ``ValueError: not enough values to unpack`` and killing the whole
    detect() call.
    """
    import arbfree_vol.arbitrage.quote_detect as qd

    surface = _two_expiry_surface(t1=0.5, sig1=0.20, t2=1.0, sig2=0.20)
    real_stv = qd.slice_total_variance

    def _empty_for_later(surface_, sl):
        if sl.expiry_time == 1.0:
            return {}
        return real_stv(surface_, sl)

    monkeypatch.setattr(qd, "slice_total_variance", _empty_for_later)

    # Must return a report, not raise ValueError
    report = detect(surface)

    assert isinstance(report, ArbitrageReport)


# ---------------------------------------------------------------------------
# Threshold-boundary sweeps: just-inside / just-outside / exact-equality
# ---------------------------------------------------------------------------


def _flat_surface(
    quotes: list[Quote],
    risk_free: float = 0.0,
    div_yield: float = 0.0,
    T: float = 1.0,
) -> VolSurface:
    """A single-slice surface with explicit r/q (defaults zero so the
    parity RHS at K=100 is exactly S - K = 0)."""
    return VolSurface(
        spot=SPOT,
        risk_free=risk_free,
        div_yield=div_yield,
        slices=[ExpirySlice(expiry_time=T, quotes=quotes)],
    )


class TestParityThreshold:
    """Put-call parity at its 0.05 fixed fallback threshold (strict >)."""

    def _c(self, price: float, bid=None, ask=None) -> Quote:
        return Quote(strike=100.0, option_type=OptionType.CALL, price=price, bid=bid, ask=ask)

    def _p(self, price: float, bid=None, ask=None) -> Quote:
        return Quote(strike=100.0, option_type=OptionType.PUT, price=price, bid=bid, ask=ask)

    def test_exact_equality_at_threshold_not_flagged(self) -> None:
        """C - P = 0.05 == threshold; the check is strict ``>`` so the
        exact boundary is NOT a violation."""
        surface = _flat_surface([self._c(5.05), self._p(5.0)])
        assert detect(surface).is_arbitrage_free

    def test_just_inside_threshold_flagged_with_metadata(self) -> None:
        """C - P = 0.06 exceeds the 0.05 threshold by an epsilon."""
        surface = _flat_surface([self._c(5.06), self._p(5.0)])
        report = detect(surface)

        assert not report.is_arbitrage_free
        v = report.violations[0]
        assert v.kind == ViolationType.PARITY
        assert v.magnitude == approx(0.06, abs=1e-9)
        assert "put-call parity off at K=100.0" in v.detail
        assert v.offending == (
            OffendingQuote(strike=100.0, expiry_time=1.0, option_type=OptionType.CALL),
            OffendingQuote(strike=100.0, expiry_time=1.0, option_type=OptionType.PUT),
        )

    def test_just_outside_threshold_not_flagged(self) -> None:
        """C - P = 0.04 is safely inside the threshold."""
        surface = _flat_surface([self._c(5.04), self._p(5.0)])
        assert detect(surface).is_arbitrage_free

    def test_bid_ask_threshold_is_half_spread_sum(self) -> None:
        """With both sides quoted, the threshold is
        max(half_spread_C + half_spread_P, 0.05) = max(2, 0.05) = 2.0:
        a residual of 3.0 flags, a residual of 0.05 does not."""
        surface = _flat_surface([
            self._c(8.0, bid=4.0, ask=6.0),
            self._p(5.0, bid=4.0, ask=6.0),
        ])
        report = detect(surface)
        parities = [v for v in report.violations if v.kind == ViolationType.PARITY]
        assert len(parities) == 1
        assert parities[0].magnitude == approx(3.0, abs=1e-9)

        surface_clean = _flat_surface([
            self._c(5.05, bid=4.0, ask=6.0),
            self._p(5.0, bid=4.0, ask=6.0),
        ])
        assert detect(surface_clean).is_arbitrage_free


class TestMonotonicityThreshold:
    """Call prices must be non-increasing: jump > 1e-4 is a violation."""

    def _surface(self, jump: float) -> VolSurface:
        return _surface([
            Quote(strike=100.0, option_type=OptionType.CALL, price=5.0),
            Quote(strike=110.0, option_type=OptionType.CALL, price=5.0 + jump),
        ])

    def test_just_inside_threshold_flagged_with_metadata(self) -> None:
        surface = self._surface(1e-4 + 1e-6)
        report = detect(surface)

        mono = [v for v in report.violations if v.kind == ViolationType.MONOTONICITY]
        assert len(mono) == 1
        assert mono[0].magnitude == approx(1e-4 + 1e-6, abs=1e-9)
        assert "call price rose from 5.0000 to 5.0001 between K=100.0 and K=110.0" in mono[0].detail
        assert mono[0].offending == (
            OffendingQuote(strike=110.0, expiry_time=1.0, option_type=OptionType.CALL),
        )

    def test_just_outside_threshold_not_flagged(self) -> None:
        assert detect(self._surface(1e-4 - 1e-6)).is_arbitrage_free

    def test_exact_equality_at_threshold_not_flagged(self) -> None:
        """jump == 1e-4; the check is strict ``>`` so the boundary passes."""
        assert detect(self._surface(1e-4)).is_arbitrage_free


class TestButterflyThreshold:
    """Call convexity: c2 - line > 1e-4 is a violation."""

    def _surface(self, middle_extra: float) -> VolSurface:
        return _surface([
            Quote(strike=90.0, option_type=OptionType.CALL, price=10.0),
            Quote(strike=100.0, option_type=OptionType.CALL, price=6.0 + middle_extra),
            Quote(strike=110.0, option_type=OptionType.CALL, price=2.0),
        ])

    def test_just_inside_threshold_flagged_with_metadata(self) -> None:
        surface = self._surface(1e-4 + 1e-6)
        report = detect(surface)

        fly = [v for v in report.violations if v.kind == ViolationType.BUTTERFLY]
        assert len(fly) == 1
        assert fly[0].magnitude == approx(1e-4 + 1e-6, abs=1e-9)
        assert "call convexity broken at K=100.0" in fly[0].detail
        assert fly[0].offending == (
            OffendingQuote(strike=100.0, expiry_time=1.0, option_type=OptionType.CALL),
        )

    def test_just_outside_threshold_not_flagged(self) -> None:
        assert detect(self._surface(1e-4 - 1e-6)).is_arbitrage_free

    def test_exact_equality_at_threshold_not_flagged(self) -> None:
        """c2 - line == 1e-4; the check is strict ``>`` so the boundary
        passes."""
        assert detect(self._surface(1e-4)).is_arbitrage_free


class TestQuoteWideSpread:
    """Direct tests for _check_wide_spread (relative spread > 0.5)."""

    def test_violation_flagged_with_metadata(self) -> None:
        # bid=5, ask=15 -> mid=10 -> spread = 1.0 > 0.5.
        sl = ExpirySlice(expiry_time=T, quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=10.0, bid=5.0, ask=15.0),
        ])
        violations: list[ArbitrageViolation] = []
        _check_wide_spread(sl, violations)

        assert len(violations) == 1
        v = violations[0]
        assert v.kind == ViolationType.WIDE_SPREAD
        assert v.magnitude == approx(1.0, abs=1e-9)
        assert "wide bid-ask spread at K=100.0" in v.detail
        assert "bid=5.0000, ask=15.0000" in v.detail
        assert v.offending == (
            OffendingQuote(strike=100.0, expiry_time=T, option_type=OptionType.CALL),
        )

    def test_pass_equality_and_missing_side(self) -> None:
        """ratio 0.2 passes, ratio == 0.5 (bid=3, ask=5) passes (strict
        ``>``), and quotes missing a bid or ask are skipped entirely."""
        sl = ExpirySlice(expiry_time=T, quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=10.0, bid=9.0, ask=11.0),
            Quote(strike=110.0, option_type=OptionType.CALL, price=4.0, bid=3.0, ask=5.0),
            Quote(strike=120.0, option_type=OptionType.CALL, price=10.0, ask=5.0),
            Quote(strike=130.0, option_type=OptionType.CALL, price=10.0, bid=5.0),
        ])
        violations: list[ArbitrageViolation] = []
        _check_wide_spread(sl, violations)
        assert violations == []

    def test_just_inside_threshold_not_flagged(self) -> None:
        """A spread of exactly ``0.5 - 1e-6`` (just inside the strict
        ``> 0.5`` threshold) must NOT be flagged as WIDE_SPREAD.

        The existing boundary tests cover ratio 0.2 (pass), equality
        0.5 (pass), and violation 1.0; this pins the just-inside side
        of the threshold so a future threshold-off-by-one regression is
        caught."""
        # (ask - bid) / mid = 0.5 - 1e-6 with mid = 10: bid/ask are
        # symmetric around mid, so the arithmetic is exact.
        half = (0.5 - 1e-6) / 2.0
        mid = 10.0
        bid = mid * (1.0 - half)
        ask = mid * (1.0 + half)
        sl = ExpirySlice(expiry_time=T, quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=10.0,
                  bid=bid, ask=ask),
        ])
        violations: list[ArbitrageViolation] = []
        _check_wide_spread(sl, violations)
        assert violations == []

    def test_wide_spread_violation_detected_by_detect(self) -> None:
        surface = _surface([
            Quote(strike=100.0, option_type=OptionType.CALL, price=10.0, bid=5.0, ask=15.0),
        ])
        report = detect(surface)
        kinds = [v.kind for v in report.violations]
        assert ViolationType.WIDE_SPREAD in kinds


class TestCalendarThreshold:
    """Cross-slice total-variance monotonicity: gap > 1e-4 is a violation."""

    def _surface(self, w2: float) -> VolSurface:
        """T=0.5 at sigma=0.40 (w=0.08) vs T=1.0 at sigma=sqrt(w2)."""
        return _two_expiry_surface(t1=0.5, sig1=0.40, t2=1.0, sig2=np.sqrt(w2))

    def test_just_inside_threshold_flagged_with_metadata(self) -> None:
        # w1 - w2 = 0.08 - 0.0789 = 1.1e-3 > 1e-4.
        surface = self._surface(0.08 - 0.0011)
        report = detect(surface)

        cal = [v for v in report.violations if v.kind == ViolationType.CALENDAR]
        assert len(cal) == 1
        assert cal[0].magnitude == approx(1.1e-3, abs=1e-5)
        assert "calendar arb: T=0.5000 > T=1.0000" in cal[0].detail

    def test_just_outside_threshold_not_flagged(self) -> None:
        # w1 - w2 = 0.08 - 0.07991 = 9e-5 < 1e-4.
        surface = self._surface(0.08 - 0.00009)
        assert detect(surface).is_arbitrage_free

    def test_exact_equality_at_threshold_not_flagged(self) -> None:
        """w1 - w2 = 1e-4 exactly; the interpolated grid gap lands on the
        safe side of the strict ``>`` comparison."""
        surface = self._surface(0.08 - 0.0001)
        report = detect(surface)
        assert all(v.kind != ViolationType.CALENDAR for v in report.violations)

    def test_three_slices_pair_attribution(self) -> None:
        """Only the (T=0.5, T=1.0) pair violates: every calendar violation
        must name that pair and none may name (T=1.0, T=2.0)."""
        surface = VolSurface(
            spot=SPOT,
            risk_free=RISK_FREE,
            div_yield=DIV_YIELD,
            slices=[
                ExpirySlice(expiry_time=0.5, quotes=[
                    _call_quote(95.0, 0.40, 0.5),
                    _call_quote(100.0, 0.40, 0.5),
                ]),
                ExpirySlice(expiry_time=1.0, quotes=[
                    _call_quote(95.0, 0.20, 1.0),
                    _call_quote(100.0, 0.20, 1.0),
                ]),
                ExpirySlice(expiry_time=2.0, quotes=[
                    _call_quote(95.0, 0.20, 2.0),
                    _call_quote(100.0, 0.20, 2.0),
                ]),
            ],
        )
        report = detect(surface)

        cal = [v for v in report.violations if v.kind == ViolationType.CALENDAR]
        assert len(cal) >= 1
        assert all("T=0.5000 > T=1.0000" in v.detail for v in cal), (
            "all calendar violations must be attributed to the violating "
            "T=0.5 -> T=1.0 pair"
        )
        assert all("T=1.0000 > T=2.0000" not in v.detail for v in cal), (
            "the clean T=1.0 -> T=2.0 pair must not be flagged"
        )
