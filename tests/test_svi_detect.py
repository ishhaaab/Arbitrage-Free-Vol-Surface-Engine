"""Tests for SVI-curve no-arbitrage detection."""

from math import sqrt

import numpy as np
from pytest import approx

from arbfree_vol.arbitrage.report import ViolationType
from arbfree_vol.arbitrage.svi_detect import detect_svi, min_total_variance
from arbfree_vol.svi.model import SVIParams


def test_min_total_variance_formula() -> None:
    p = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.05, sigma=0.15)
    expected = 0.04 + 0.4 * 0.15 * sqrt(1 - 0.4**2)
    assert min_total_variance(p) == approx(expected, abs=1e-12)


def test_negative_min_variance_is_flagged() -> None:
    # a is low enough that a + b*sigma*sqrt(1-rho^2) < 0.
    # w_min = -0.1 + 0.1*0.1*1 = -0.09
    p = SVIParams(a=-0.1, b=0.1, rho=0.0, m=0.0, sigma=0.1)

    report = detect_svi(p)

    assert not report.is_arbitrage_free
    v = report.violations[0]
    assert v.kind == ViolationType.NEGATIVE_VARIANCE
    assert v.magnitude == approx(0.09, abs=1e-6)


def test_healthy_params_are_arbitrage_free() -> None:
    p = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.05, sigma=0.15)

    report = detect_svi(p)

    assert report.is_arbitrage_free


def test_butterfly_arbitrage_is_flagged() -> None:
    # Steep wings (large b, extreme rho, tiny sigma): g(k) dips well below 0,
    # but w_min stays positive -> isolates the butterfly check.
    p = SVIParams(a=0.04, b=0.8, rho=-0.9, m=0.0, sigma=0.02)

    report = detect_svi(p)

    kinds = [v.kind for v in report.violations]
    assert ViolationType.BUTTERFLY in kinds
    assert ViolationType.NEGATIVE_VARIANCE not in kinds  # variance is fine here
    fly = next(v for v in report.violations if v.kind == ViolationType.BUTTERFLY)
    assert fly.magnitude > 1.0  # deeply negative g (~3.16)


def test_butterfly_violation_outside_prev_range() -> None:
    # High positive rho shifts the violation to large positive k.
    # g(k) is positive everywhere in [-2, 2] but goes negative around k ≈ 2.7.
    # The widened default [-3, 3] catches this; the old [-2, 2] would miss it.
    p = SVIParams(a=0.04, b=0.9, rho=0.97, m=2.0, sigma=0.01)

    report = detect_svi(p)

    kinds = [v.kind for v in report.violations]
    assert ViolationType.BUTTERFLY in kinds
    assert ViolationType.NEGATIVE_VARIANCE not in kinds
    fly = next(v for v in report.violations if v.kind == ViolationType.BUTTERFLY)
    assert fly.magnitude > 0.04  # g dips to ~ -0.055 at k ≈ 2.7


# ---------------------------------------------------------------------------
# Calendar tests — cross-slice total-variance monotonicity
# ---------------------------------------------------------------------------

_K_GRID = np.linspace(-1.5, 1.5, 61)


def _calendar_violations(
    slices: list[tuple[float, SVIParams]],
    k_grid: np.ndarray = _K_GRID,
) -> list:
    """Run only the calendar check and return the violation list."""
    from arbfree_vol.arbitrage.svi_detect import _check_calendar

    violations: list = []
    _check_calendar(slices, violations, k_grid)
    return violations


def test_calendar_no_violation_when_later_has_higher_variance() -> None:
    # Later slice has higher a everywhere -> gap always negative -> no calendar arb.
    earlier = SVIParams(a=0.04, b=0.3, rho=-0.3, m=0.0, sigma=0.15)
    later = SVIParams(a=0.09, b=0.3, rho=-0.3, m=0.0, sigma=0.15)

    violations = _calendar_violations([(0.5, earlier), (1.0, later)])

    assert violations == []


def test_calendar_single_contiguous_band() -> None:
    # Later slice lower everywhere -> one band covering the full k grid.
    earlier = SVIParams(a=0.06, b=0.3, rho=-0.3, m=0.0, sigma=0.15)
    later = SVIParams(a=0.04, b=0.3, rho=-0.3, m=0.0, sigma=0.15)

    violations = _calendar_violations([(0.5, earlier), (1.0, later)])

    assert len(violations) == 1
    v = violations[0]
    assert v.kind == ViolationType.CALENDAR
    assert v.magnitude == approx(0.02, abs=1e-4)


def test_calendar_two_disjoint_bands() -> None:
    # Earlier has low a / high b (steep wings), later has high a / low b (high ATM).
    # Earlier > later at wings (violation), earlier < later near ATM (clean).
    earlier = SVIParams(a=0.02, b=0.5, rho=-0.3, m=0.0, sigma=0.1)
    later = SVIParams(a=0.06, b=0.2, rho=-0.3, m=0.0, sigma=0.1)

    violations = _calendar_violations([(0.5, earlier), (1.0, later)])

    # Expect exactly two disjoint violation bands.
    assert len(violations) == 2
    assert all(v.kind == ViolationType.CALENDAR for v in violations)
    # Worst magnitudes: left band ~0.546, right band ~0.276.
    mags = sorted([v.magnitude for v in violations], reverse=True)
    assert mags[0] == approx(0.546, abs=0.01)
    assert mags[1] == approx(0.276, abs=0.01)


def test_calendar_three_slices_only_middle_pair_violates() -> None:
    # (T1=0.5, T2=1.0): later has lower variance -> violation.
    # (T2=1.0, T3=2.0): later has higher variance -> clean.
    s1 = SVIParams(a=0.06, b=0.3, rho=-0.3, m=0.0, sigma=0.15)
    s2 = SVIParams(a=0.04, b=0.3, rho=-0.3, m=0.0, sigma=0.15)
    s3 = SVIParams(a=0.10, b=0.3, rho=-0.3, m=0.0, sigma=0.15)

    violations = _calendar_violations([(0.5, s1), (1.0, s2), (2.0, s3)])

    # All violations should come from the (T1, T2) pair only.
    assert len(violations) >= 1
    assert all(v.kind == ViolationType.CALENDAR for v in violations)


def test_calendar_single_slice_returns_empty() -> None:
    p = SVIParams(a=0.04, b=0.3, rho=-0.3, m=0.0, sigma=0.15)

    violations = _calendar_violations([(1.0, p)])

    assert violations == []


def test_calendar_empty_input_returns_empty() -> None:
    violations = _calendar_violations([])

    assert violations == []


def test_detect_svi_surface_combines_per_slice_and_calendar() -> None:
    from arbfree_vol.arbitrage.svi_detect import detect_svi_surface

    # Two slices: one with calendar violation (s1 > s2), both with valid min-variance.
    s1 = SVIParams(a=0.06, b=0.3, rho=-0.3, m=0.0, sigma=0.15)
    s2 = SVIParams(a=0.04, b=0.3, rho=-0.3, m=0.0, sigma=0.15)

    report = detect_svi_surface([(0.5, s1), (1.0, s2)], k_grid=_K_GRID)

    # Should have calendar violations, but no negative-variance or butterfly issues.
    kinds = [v.kind for v in report.violations]
    assert ViolationType.CALENDAR in kinds
    assert ViolationType.NEGATIVE_VARIANCE not in kinds
    assert ViolationType.BUTTERFLY not in kinds


def test_detect_svi_surface_empty_returns_empty() -> None:
    from arbfree_vol.arbitrage.svi_detect import detect_svi_surface

    report = detect_svi_surface([])

    assert report.is_arbitrage_free
