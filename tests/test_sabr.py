"""Tests for the SABR model and calibration."""

import numpy as np
import pytest
from pytest import approx

from arbfree_vol.sabr.model import (
    SABRParams,
    clamp_to_sabr_domain,
    sabr_implied_vol,
    sabr_total_variance,
    to_raw_svi_params,
)
from arbfree_vol.sabr.calibration import calibrate_sabr
from arbfree_vol.svi.model import svi_total_variance

# Recovery tolerance for SABR calibration (beta is fixed, so recovery is
# approximate)
_SABR_RECOVERY_TOL = 0.02

# NOTE: these parameters are a repo FIXTURE. Hagan et al. (2002) 'Managing Smile Risk' contains
# no numeric worked example with these values; the paper's concrete numerics are the swaption
# vol-of-vol/correlation tables. The fixture is used for consistency tests only.
# A reproducible SABR parameter set
_ALPHA = 0.2
_BETA = 0.5
_RHO = -0.3
_NU = 0.4
_F = 100.0
_T = 1.0


def test_sabr_atm_limit_consistent() -> None:
    """Very near ATM, sabr_implied_vol must match the explicit ATM closed form."""
    k_atm = 1e-10  # near zero
    sigma_near = sabr_implied_vol(k_atm, _F, _T, _ALPHA, _BETA, _RHO, _NU)

    # Explicit ATM formula when k=0
    F_1mb = _F ** (1.0 - _BETA)
    sigma_atm = _ALPHA / F_1mb * (
        1.0 + _T * (
            ((1.0 - _BETA) ** 2 / 24.0) * _ALPHA ** 2 / (_F ** (2.0 - 2.0 * _BETA))
            + (_RHO * _BETA * _ALPHA * _NU) / (4.0 * F_1mb)
            + (2.0 - 3.0 * _RHO * _RHO) * _NU * _NU / 24.0
        )
    )

    assert sigma_near == approx(sigma_atm, abs=1e-8)


def test_sabr_calibration_recovers_known_params() -> None:
    """Calibration of SABR parameters approximately recovers known values."""
    ks = np.linspace(-0.5, 0.5, 11)
    points = [
        (float(k), sabr_total_variance(float(k), _F, _T, _ALPHA, _BETA, _RHO, _NU))
        for k in ks
    ]

    fitted = calibrate_sabr(points, forward=_F, expiry_time=_T, beta_hint=_BETA)

    # SABR calibration is approximate due to fixed beta; use wider tolerance
    assert fitted.alpha == approx(_ALPHA, abs=_SABR_RECOVERY_TOL)
    assert fitted.rho == approx(_RHO, abs=_SABR_RECOVERY_TOL)
    assert fitted.nu == approx(_NU, abs=_SABR_RECOVERY_TOL)
    assert fitted.beta == _BETA  # beta_hint is fixed


def test_sabr_calibrated_curve_fits_the_cloud() -> None:
    """Pointwise fit of calibrated curve must be tight."""
    ks = np.linspace(-0.5, 0.5, 11)
    points = [
        (float(k), sabr_total_variance(float(k), _F, _T, _ALPHA, _BETA, _RHO, _NU))
        for k in ks
    ]

    fitted = calibrate_sabr(points, forward=_F, expiry_time=_T, beta_hint=_BETA)

    for k, w in points:
        w_fit = sabr_total_variance(
            float(k), _F, _T,
            fitted.alpha, fitted.beta, fitted.rho, fitted.nu
        )
        assert w_fit == approx(w, abs=1e-6)


def test_sabr_calibrate_raises_with_too_few_points() -> None:
    """Calibration must raise ValueError with fewer than 5 points."""
    with pytest.raises(ValueError):
        calibrate_sabr([(0.0, 0.04), (0.1, 0.05), (0.2, 0.06)],
                        forward=_F, expiry_time=_T)


def test_to_raw_svi_params_returns_valid_svi() -> None:
    """The SABR -> raw SVI adapter must produce plausible SVI parameters."""
    sabr_params = SABRParams(alpha=_ALPHA, beta=_BETA, rho=_RHO, nu=_NU)
    a, b, r, m, sigma = to_raw_svi_params(sabr_params, _F, _T)

    # Check basic SVI parameter sanity
    assert b >= 0
    assert -1.0 < r < 1.0
    assert sigma > 0

    # At k=0 the mapped SVI should approximately reproduce SABR total variance
    w_sabr_0 = sabr_total_variance(0.0, _F, _T, _ALPHA, _BETA, _RHO, _NU)
    w_svi_0 = svi_total_variance(0.0, a, b, r, m, sigma)
    assert w_svi_0 == approx(w_sabr_0, abs=0.01)


def _sabr_atm_closed_form(alpha: float, beta: float, rho: float,
                           nu: float, F: float, T: float) -> float:
    """Explicit SABR ATM closed-form (Hagan et al. 2002)."""
    F_1mb = F ** (1.0 - beta)
    F_2mb = F ** (2.0 - 2.0 * beta)
    sigma_atm = alpha / F_1mb
    corr = (
        ((1.0 - beta) ** 2 / 24.0) * alpha ** 2 / F_2mb
        + (rho * beta * alpha * nu) / (4.0 * F_1mb)
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    )
    return sigma_atm * (1.0 + corr * T)


def test_sabr_log_moneyness_bracket_known_values() -> None:
    """Regression against Hagan et al. (2002) Eq 2.17a including the
    leading log-moneyness bracket.

    The Eq 2.17a structure is verified verbatim against Hagan et al.
    (2002): the leading bracket
    ``1 + (1-beta)^2/24 log^2(F/K) + (1-beta)^4/1920 log^4(F/K)`` is
    confirmed present at Eq 2.17a of the paper.

    The numeric reference values below were computed from that formula
    for F=100, T=1, alpha=0.25, beta=0.5, rho=-0.4, nu=0.8.  They are
    self-consistency references (computed via the repo's own formula),
    pending an independent oracle cross-check in Phase 2
    (py_vollib/pysabr).  Keep them as regression pins.

    Note: per Obloj (2008, arXiv:0708.0998, footnote 4), the simplified
    formula 2.17a = (A.69c) is NOT affected by Obloj's correction (which
    targets the general formula A.65 for beta<1), so these pins are
    in-scope.

    Without the bracket, the wing IV is understated by ~0.065% at
    |k|=0.25 and ~0.58% at k=0.75.
    """
    F, T = 100.0, 1.0
    alpha, beta, rho, nu = 0.25, 0.5, -0.4, 0.8
    cases = [
        (0.0, 0.025988496094),
        (0.25, 0.063028392718),
        (-0.25, 0.085651314868),
        (0.75, 0.137731503445),
    ]
    for k, expected in cases:
        iv = sabr_implied_vol(k, F, T, alpha, beta, rho, nu)
        assert iv == approx(expected, abs=1e-10), (
            f"sabr_implied_vol({k}) = {iv:.12f}, expected {expected:.12f}"
        )


def test_sabr_rho_zero_symmetry() -> None:
    """With rho=0 and beta=1 the SABR smile is symmetric in k; with non-zero
    rho (beta=0.5) it is not.  (Beta != 1 introduces asymmetry through the
    FK_pow factor even when rho=0, so we set beta=1 for the symmetry check.)"""
    ks = np.linspace(-0.3, 0.3, 13)
    for k in ks:
        pos = sabr_implied_vol(float(k), _F, _T, _ALPHA, 1.0, 0.0, _NU)
        neg = sabr_implied_vol(-float(k), _F, _T, _ALPHA, 1.0, 0.0, _NU)
        assert pos == approx(neg, abs=1e-10)

    # Default params (beta=0.5, rho=-0.3) give a skewed smile
    pos_asym = sabr_implied_vol(0.3, _F, _T, _ALPHA, _BETA, _RHO, _NU)
    neg_asym = sabr_implied_vol(-0.3, _F, _T, _ALPHA, _BETA, _RHO, _NU)
    assert abs(pos_asym - neg_asym) > 1e-6


def test_sabr_atm_consistency_across_param_sets() -> None:
    """sabr_implied_vol at k≈0 matches the ATM closed form for different regimes."""
    param_sets = [
        (0.2, 0.5, -0.3, 0.4, 100.0, 1.0),  # standard
        (0.35, 0.9, -0.5, 0.8, 50.0, 2.0),  # rates-like
        (0.15, 0.0, 0.2, 0.3, 200.0, 0.5),  # normal-like
    ]
    for alpha, beta, rho, nu, F, T in param_sets:
        imp_vol = sabr_implied_vol(1e-10, F, T, alpha, beta, rho, nu)
        atm_cf = _sabr_atm_closed_form(alpha, beta, rho, nu, F, T)
        assert imp_vol == approx(atm_cf, abs=1e-8)


def test_sabr_smile_positive() -> None:
    """SABR implied vol must remain positive across a realistic smile."""
    ks = np.linspace(-0.5, 0.5, 21)
    for k in ks:
        iv = sabr_implied_vol(float(k), _F, _T, _ALPHA, _BETA, _RHO, _NU)
        assert iv > 0, f"Non-positive IV at k={k}: {iv}"


def test_clamp_to_sabr_domain_boundary_inputs() -> None:
    """The domain clamp nudges boundary values inside the model and leaves
    in-domain values untouched.

    The margins are derived from the SABRParams field constraints
    (``gt``/``lt``/``ge``/``le`` metadata), so the clamp cannot drift from
    the model bounds.  Exclusive bounds (rho, alpha, nu) get a 1e-9
    nudge; inclusive bounds (beta in [0, 1]) clamp to the bound exactly.
    """
    # rho: exactly-on-bound must be nudged strictly inside (-0.999, 0.999).
    r_hi = clamp_to_sabr_domain("rho", 0.999)
    assert r_hi < 0.999
    assert r_hi == approx(0.999 - 1e-9)
    r_lo = clamp_to_sabr_domain("rho", -0.999)
    assert r_lo > -0.999
    assert r_lo == approx(-0.999 + 1e-9)

    # In-domain values pass through unchanged (tight clamp).
    assert clamp_to_sabr_domain("rho", 0.5) == 0.5
    assert clamp_to_sabr_domain("rho", -0.998) == -0.998

    # alpha / nu: gt=0 — boundary and below-bound land above 0.
    assert clamp_to_sabr_domain("alpha", 0.0) > 0
    assert clamp_to_sabr_domain("nu", 0.0) > 0
    assert clamp_to_sabr_domain("alpha", 0.2) == 0.2

    # beta: inclusive [0, 1] — clamps to the bound, not nudged past it.
    assert clamp_to_sabr_domain("beta", -0.1) == 0.0
    assert clamp_to_sabr_domain("beta", 1.5) == 1.0
    assert clamp_to_sabr_domain("beta", 0.5) == 0.5

    # The clamped boundary values construct valid SABRParams.
    p = SABRParams(alpha=clamp_to_sabr_domain("alpha", 0.0),
                   beta=clamp_to_sabr_domain("beta", 1.5),
                   rho=clamp_to_sabr_domain("rho", 0.999),
                   nu=clamp_to_sabr_domain("nu", 0.0))
    assert p.rho < 0.999
    assert p.alpha > 0 and p.nu > 0
