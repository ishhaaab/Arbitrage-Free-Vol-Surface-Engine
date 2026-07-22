"""Benchmark regression tests for raw SVI parameterization.

Uses the classic Gatheral (2004) example parameter set to verify
known-value and no-arbitrage properties.
"""

from math import sqrt

import numpy as np
from pytest import approx

from arbfree_vol.arbitrage.svi_detect import detect_svi, min_total_variance
from arbfree_vol.svi.model import SVIParams, svi_total_variance


# Gatheral (2004) canonical parameter set
TRUE = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.05, sigma=0.15)


def test_benchmark_svi_min_variance_positive() -> None:
    """The benchmark parameters must yield strictly positive min total variance."""
    w_min = min_total_variance(TRUE)
    assert w_min > 0
    expected = 0.04 + 0.4 * 0.15 * sqrt(1 - 0.4 ** 2)
    assert w_min == approx(expected, abs=1e-12)


def test_benchmark_svi_atm_value() -> None:
    """At k=m, total variance = a + b * sigma (benchmark reference)."""
    w = svi_total_variance(TRUE.m, TRUE.a, TRUE.b, TRUE.rho, TRUE.m, TRUE.sigma)
    expected = TRUE.a + TRUE.b * TRUE.sigma
    assert w == approx(expected, abs=1e-12)


def test_benchmark_svi_no_butterfly_arb_on_grid() -> None:
    """The standard Gatheral example must pass the no-arb check."""
    report = detect_svi(TRUE)
    assert report.is_arbitrage_free


def test_benchmark_svi_asymptotic_slope() -> None:
    """At large |k|, SVI wings match the asymptotic slope b*(1+rho) and b*(1-rho).

    Right wing (k → +∞): w ≈ a + b·(1+ρ)·(k−m)
    Left wing  (k → −∞): w ≈ a + b·(1−ρ)·|k−m|
    (for k < m, |k-m| = -(k-m) in the left-wing expression)
    """
    # Use k=100 to ensure the asymptotic approximation is within rel=1e-4
    # (the O(1/k) correction b*sigma^2/(2*(k-m)) at k=10 is ~1.86e-4 rel,
    #  exceeding rel=1e-4, but at k=100 it drops to ~1.9e-6).
    k_pos = 100.0
    w = svi_total_variance(k_pos, TRUE.a, TRUE.b, TRUE.rho, TRUE.m, TRUE.sigma)
    expected_slope = TRUE.b * (1.0 + TRUE.rho)  # = 0.4 * 0.6 = 0.24
    assert w == approx(TRUE.a + expected_slope * (k_pos - TRUE.m), rel=1e-4)

    k_neg = -100.0
    w_left = svi_total_variance(k_neg, TRUE.a, TRUE.b, TRUE.rho, TRUE.m, TRUE.sigma)
    expected_slope_left = TRUE.b * (1.0 - TRUE.rho)  # = 0.4 * 1.4 = 0.56
    # Left wing uses |k-m| = -(k-m) as k-m < 0
    assert w_left == approx(TRUE.a + expected_slope_left * abs(k_neg - TRUE.m), rel=1e-4)
