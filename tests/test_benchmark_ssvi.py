"""Benchmark regression tests for SSVI parameterization.

Verifies the Gatheral-Jacquier (2014) no-arb condition across a range
of parameter tuples, and confirms the defining SSVI property: w(0) = theta.
"""

from math import isfinite, sqrt

import numpy as np
from pytest import approx

from arbfree_vol.ssvi.model import gatheral_jacquier_condition, ssvi_w


def _safe_psi(theta: float, rho: float, fraction: float = 0.5) -> float:
    """Return psi safely inside BOTH Gatheral-Jacquier bounds.

    The two bounds are ``theta * psi * (1 + |rho|) <= 4`` and
    ``theta * psi^2 * (1 + |rho|) <= 4``, so the largest safe psi is
    ``min(4 / (theta * (1 + |rho|)), sqrt(4 / (theta * (1 + |rho|))))``.
    """
    psi_bound_1 = 4.0 / (theta * (1.0 + abs(rho)))
    psi_bound_2 = sqrt(4.0 / (theta * (1.0 + abs(rho))))
    return fraction * min(psi_bound_1, psi_bound_2)


def test_benchmark_ssvi_gj_range() -> None:
    """For a grid of (theta, rho, psi) tuples safely inside GJ bound,
    the residual must be positive; at 1.2x the bound it must be negative."""
    thetas = [0.04, 0.1, 0.2, 0.4]
    rhos = [-0.4, -0.2, 0.0, 0.3]

    for theta in thetas:
        for rho in rhos:
            psi_safe = _safe_psi(theta, rho, 0.5)
            # Safely inside bound
            r_safe = gatheral_jacquier_condition(theta, rho, psi_safe)
            assert r_safe > 0, (
                f"theta={theta}, rho={rho}, psi={psi_safe}: "
                f"expected positive residual, got {r_safe}"
            )

            # Beyond the GJ bound → negative residual
            psi_unsafe = _safe_psi(theta, rho, 1.2)
            r_unsafe = gatheral_jacquier_condition(theta, rho, psi_unsafe)
            assert r_unsafe < 0, (
                f"theta={theta}, rho={rho}, psi={psi_unsafe}: "
                f"expected negative residual, got {r_unsafe}"
            )


def test_benchmark_ssvi_smile_positive() -> None:
    """SSVI total variance must be positive for all k in [-1, 1]."""
    theta, rho = 0.04, -0.4
    psi = _safe_psi(theta, rho, 0.5)

    ks = np.linspace(-1.0, 1.0, 21)
    for k in ks:
        w = ssvi_w(k, theta, rho, psi)
        assert w > 0, f"Non-positive total variance at k={k}: w={w}"


def test_benchmark_ssvi_atm() -> None:
    """At k=0, SSVI total variance must equal theta for any valid params."""
    params_list = [
        (0.04, -0.4, 0.5),
        (0.10, 0.0, 1.0),
        (0.25, 0.3, 2.0),
        (0.50, -0.2, 0.8),
    ]
    for theta, rho, psi in params_list:
        w0 = ssvi_w(0.0, theta, rho, psi)
        assert w0 == approx(theta, abs=1e-12)


def test_benchmark_ssvi_rho_zero_symmetry() -> None:
    """With rho=0 the SSVI smile is symmetric in k; with non-zero rho it is not."""
    theta, rho, psi = 0.04, 0.0, 0.5
    ks = np.linspace(-1.0, 1.0, 21)
    for k in ks:
        w_pos = ssvi_w(float(k), theta, rho, psi)
        w_neg = ssvi_w(-float(k), theta, rho, psi)
        assert w_pos == approx(w_neg, abs=1e-12)

    # With non-zero rho the smile is NOT symmetric
    rho_asym = -0.4
    w_pos = ssvi_w(0.5, theta, rho_asym, psi)
    w_neg = ssvi_w(-0.5, theta, rho_asym, psi)
    assert abs(w_pos - w_neg) > 1e-6


def test_ssvi_w0_theta_identity_nonatm_k() -> None:
    """SSVI definition (Gatheral & Jacquier 2014, Definition 4.1):
    w(0, theta) = theta exactly.  For k != 0, w(k, theta) must be finite
    and positive for representative (theta, rho, psi).

    Uses a distinct parameter grid from ``test_benchmark_ssvi_atm`` to
    broaden coverage, plus an explicit non-ATM finiteness/positivity
    check that the ATM-only test does not exercise.
    """
    params_list = [
        (0.02, -0.5, 0.7),
        (0.06, 0.2, 1.5),
        (0.30, -0.1, 0.9),
        (0.80, 0.4, 0.6),
    ]
    ks = np.linspace(-1.0, 1.0, 21)
    for theta, rho, psi in params_list:
        w0 = ssvi_w(0.0, theta, rho, psi)
        assert w0 == approx(theta, abs=1e-12)
        for k in ks:
            w = ssvi_w(float(k), theta, rho, psi)
            assert isfinite(w), f"non-finite w at k={k} for (theta,rho,psi)={(theta, rho, psi)}"
            assert w > 0, f"non-positive w at k={k} for (theta,rho,psi)={(theta, rho, psi)}"
