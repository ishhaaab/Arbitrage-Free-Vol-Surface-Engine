"""Tests for the SABR model and calibration."""

import numpy as np
import pytest
from pytest import approx

from arbfree_vol.sabr.model import (
    SABRParams,
    sabr_implied_vol,
    sabr_total_variance,
    to_raw_svi_params,
)
from arbfree_vol.sabr.calibration import calibrate_sabr
from arbfree_vol.svi.model import svi_total_variance

# Recovery tolerance for SABR calibration (beta is fixed, so recovery is
# approximate)
_SABR_RECOVERY_TOL = 0.02

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
