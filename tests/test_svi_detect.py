"""Tests for SVI-curve no-arbitrage detection."""

from math import sqrt

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
