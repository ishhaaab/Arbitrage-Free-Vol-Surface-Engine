"""Benchmark regression tests for raw SVI parameterization.

Verifies the no-arbitrage and known-value identities that hold for ANY
valid SVI parameter set.  The module-level ``TRUE`` fixture is a
repo-internal parameter set (see the comment below), NOT a paper value.
The reconstructed Gatheral (2004) base case is exercised only through
algebraic self-consistency identities — ``test_benchmark_svi_gatheral2004_identity_checks``
— which are NOT independently verified published outputs.
"""

from math import sqrt

import numpy as np
from pytest import approx

from arbfree_vol.arbitrage.svi_detect import detect_svi, min_total_variance
from arbfree_vol.svi.model import SVIParams, svi_total_variance
from arbfree_vol.ssvi.model import essvi_psi, gatheral_jacquier_condition


# FIXTURE parameter set — NOT from any paper. The reconstructed Gatheral (2004) base case is
# (a=0.04, b=0.4, rho=-0.4, sigma=0.1, m=0); the tuple below (sigma=0.15, m=0.05) was
# a repo-internal choice. Used only to exercise property tests (identities hold for ANY valid SVI).
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
    """The repo fixture must pass the no-arb check."""
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


def test_benchmark_svi_gatheral2004_identity_checks() -> None:
    """Self-consistency identities for the reconstructed Gatheral (2004)
    base case (a=0.04, b=0.4, rho=-0.4, sigma=0.1, m=0).

    IMPORTANT: these are ALGEBRAIC IDENTITY checks derived from the
    parameter tuple itself (the closed-form min-total-variance value, the
    ATM value a + b*sigma, and the no-butterfly-arb grid check).  They
    are NOT independently verified published outputs — no second source
    was cross-checked, so they certify internal consistency only.
    """
    base = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.0, sigma=0.1)

    w_min = min_total_variance(base)
    expected = 0.04 + 0.4 * 0.1 * sqrt(1 - 0.4 ** 2)
    assert w_min == approx(expected, abs=1e-12)

    w_atm = svi_total_variance(base.m, base.a, base.b, base.rho, base.m, base.sigma)
    assert w_atm == approx(base.a + base.b * base.sigma, abs=1e-12)

    report = detect_svi(base)
    assert report.is_arbitrage_free


def test_gj2014_vogt_example_is_butterfly_arbitrageable() -> None:
    """Gatheral & Jacquier (2014), 'Arbitrage-free SVI volatility
    surfaces' (arXiv:1204.0646), Example 3.1: the raw SVI parameter
    tuple (a,b,m,rho,sigma) = (-0.0410, 0.1331, 0.3586, 0.3060, 0.4153)
    with t=1 — the 'Axel Vogt on wilmott.com' example — produces a smile
    with a negative risk-neutral density (butterfly arbitrage).

    The violation is detected on the default ±3.0 grid
    (g ≈ -0.033 at k ≈ 0.88).
    """
    vogt = SVIParams(a=-0.0410, b=0.1331, rho=0.3060, m=0.3586, sigma=0.4153)
    report = detect_svi(vogt)
    assert not report.is_arbitrage_free


def test_gj2014_ssvi_power_law_identities() -> None:
    """Gatheral & Jacquier (2014) Example 4.2 / Eq 4.2: the eSSVI wing
    function is the power law phi(theta) = eta * theta^(-gamma) with
    eta > 0, 0 < gamma < 1.

    (a) essvi_psi(theta, eta, gamma) == eta * theta**(-gamma) for several
        (theta, eta, gamma) triples.
    (b) essvi_w(k, theta, rho, eta, gamma) == ssvi_w(k, theta, rho, psi)
        with psi = essvi_psi(theta, eta, gamma) — this is already covered
        by test_ssvi_eSSVI_consistency in tests/test_ssvi.py, so it is not
        duplicated here.
    """
    cases = [
        (0.04, 0.5, 0.5),
        (0.16, 1.0, 0.3),
        (0.01, 0.2, 0.7),
    ]
    for theta, eta, gamma in cases:
        assert essvi_psi(theta, eta, gamma) == approx(eta * theta ** (-gamma), abs=1e-12)


def test_gj2014_theorem42_boundary_distinction() -> None:
    """Gatheral & Jacquier (2014) Theorem 4.2: butterfly-sufficiency for
    SSVI requires BOTH
        theta * phi(theta) * (1 + |rho|) < 4      (strict)
        theta * phi(theta)^2 * (1 + |rho|) <= 4   (non-strict)

    The repo's gatheral_jacquier_condition returns
    min(4 - theta*psi*(1+|rho|), 4 - theta*psi^2*(1+|rho|)) and treats
    residual >= 0 as 'butterfly-safe'.

    NOTE (semantic mismatch): the default mode (strict=False) folds the
    paper's strict condition 1 and non-strict condition 2 into a single
    residual >= 0 check, so the strict/non-strict distinction is LOST: a
    case where condition 1 holds with equality but condition 2 holds
    strictly (e.g. theta=8, rho=0, psi=0.5 -> residual exactly 0) is
    reported 'safe' although the paper's strict condition 1 says it is
    NOT safe.

    STRICT MODE (strict=True): condition 1 is enforced STRICTLY per the
    paper — the equality-only case returns a NEGATIVE residual (not
    safe) while the healthy case still returns positive.
    """
    # Boundary case: theta*psi*(1+|rho|) == 4 exactly AND the psi^2 bound
    # also fails (16 > 4), so the min-residual is clearly negative -> not
    # butterfly-safe.  theta=1.0, rho=0.0, psi=4.0:
    #   first  residual = 4 - 1*4*(1+0)   = 0
    #   second residual = 4 - 1*16*(1+0)  = -12
    boundary = gatheral_jacquier_condition(1.0, 0.0, 4.0)
    assert boundary == approx(-12.0, abs=1e-12)
    assert boundary < 0  # NOT butterfly-safe

    # Both conditions hold strictly: theta*psi*(1+|rho|) = 0.028 < 4 and
    # theta*psi^2*(1+|rho|) = 0.014 <= 4.
    safe = gatheral_jacquier_condition(0.04, -0.4, 0.5)
    assert safe == approx(3.972, abs=1e-3)
    assert safe >= 0  # butterfly-safe

    # Documented strict/non-strict mismatch: condition-1 equality with
    # condition-2 strict.  theta=8, rho=0, psi=0.5:
    #   first  = 4 - 8*0.5*(1+0)  = 0   (== 4 boundary, paper says NOT safe)
    #   second = 4 - 8*0.25*(1+0) = 2   (< 4, safe)
    # The repo reports residual 0 (>= 0 -> 'safe'); the paper's strict
    # condition 1 says NOT safe.  We assert the repo's actual behaviour.
    equality_only = gatheral_jacquier_condition(8.0, 0.0, 0.5)
    assert equality_only == approx(0.0, abs=1e-12)

    # STRICT MODE: the same equality-only case is now a violation — the
    # residual must be negative (nudged below the strict-1 epsilon).
    equality_only_strict = gatheral_jacquier_condition(8.0, 0.0, 0.5, strict=True)
    assert equality_only_strict < 0, (
        f"strict mode must flag condition-1 equality as not safe, got "
        f"{equality_only_strict}"
    )

    # STRICT MODE healthy case: condition 1 holds strictly (3.972 > 0),
    # so the residual stays positive and identical to the default mode.
    safe_strict = gatheral_jacquier_condition(0.04, -0.4, 0.5, strict=True)
    assert safe_strict >= 0  # butterfly-safe
    assert safe_strict == approx(safe, abs=1e-12)

    # STRICT MODE boundary case: both bounds fail anyway, so the residual
    # is still the clearly-negative second bound.
    boundary_strict = gatheral_jacquier_condition(1.0, 0.0, 4.0, strict=True)
    assert boundary_strict == approx(-12.0, abs=1e-12)
    assert boundary_strict < 0
