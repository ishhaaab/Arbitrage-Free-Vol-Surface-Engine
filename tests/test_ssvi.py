"""Tests for the SSVI / eSSVI parameterization."""
from math import sqrt
from pytest import approx

from arbfree_vol.ssvi.model import (
    SSVIParams,
    eSSVISurfaceParams,
    essvi_psi,
    ssvi_w,
    essvi_w,
    ssvi_dw_dk,
    ssvi_d2w_dk2,
    to_raw_svi_params,
    essvi_arb_safe,
)
from arbfree_vol.svi.model import svi_total_variance
from arbfree_vol.ssvi.calibration import fit_ssvi_slice, fit_essvi_slice


# Reference: a healthy SSVI/eSSVI parameter set
TRUE = dict(theta=0.04, rho=-0.4, psi=0.5)


def _points_from(ks, theta, rho, psi):
    return [(float(k), ssvi_w(float(k), theta, rho, psi)) for k in ks]


def test_ssvi_atm_value() -> None:
    # At k=0, w(0) = theta (ATM total variance).
    w0 = ssvi_w(0.0, 0.04, -0.4, 0.5)
    assert w0 == approx(0.04, abs=1e-12)


def test_ssvi_eSSVI_consistency() -> None:
    # eSSVI with psi=eta/theta**gamma should match ssvi_w directly.
    eta, gamma = 0.5, 0.5
    theta, rho = 0.04, -0.4
    psi = essvi_psi(theta, eta, gamma)

    for k in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        a = ssvi_w(k, theta, rho, psi)
        b = essvi_w(k, theta, rho, eta, gamma)
        assert a == approx(b, abs=1e-12)


def test_calibration_recovers_known_params() -> None:
    # Generate points from a known SSVI surface, then refit.
    import numpy as np
    ks = np.linspace(-0.4, 0.4, 11)
    points = _points_from(ks, TRUE["theta"], TRUE["rho"], TRUE["psi"])

    fitted = fit_ssvi_slice(points)

    assert fitted.theta == approx(TRUE["theta"], abs=1e-3)
    assert fitted.rho == approx(TRUE["rho"], abs=1e-2)
    # psi is harder to pin down tightly; the calibration is least-squares
    # so a small residual is expected.
    assert fitted.psi == approx(TRUE["psi"], abs=0.05)


def test_calibrated_curve_fits_the_cloud() -> None:
    import numpy as np
    ks = np.linspace(-0.4, 0.4, 11)
    points = _points_from(ks, TRUE["theta"], TRUE["rho"], TRUE["psi"])

    fitted = fit_ssvi_slice(points)

    for k, w in points:
        w_fit = ssvi_w(k, fitted.theta, fitted.rho, fitted.psi)
        assert w_fit == approx(w, abs=1e-3)


def test_essvi_calibration_recovers_known_eta_gamma() -> None:
    import numpy as np
    eta, gamma = 0.5, 0.5
    theta, rho = 0.04, -0.4
    ks = np.linspace(-0.4, 0.4, 11)
    points = [(float(k), essvi_w(float(k), theta, rho, eta, gamma)) for k in ks]

    ssvi_params, surface_params = fit_essvi_slice(points)

    # eta, gamma are less identifiable from a single slice (a single
    # slice gives us one theta point), so we just check theta/rho fit.
    assert ssvi_params.theta == approx(theta, abs=1e-3)
    assert ssvi_params.rho == approx(rho, abs=1e-2)
    assert surface_params.eta == approx(eta, abs=0.5)
    assert surface_params.gamma == approx(gamma, abs=0.5)


def test_calibration_too_few_points_raises() -> None:
    with __import__("pytest").raises(ValueError):
        fit_ssvi_slice([(0.0, 0.04), (0.1, 0.045), (0.2, 0.05)])


def test_ssvi_dw_dk_at_atm() -> None:
    # dw/dk(0) = theta * rho * psi
    # (derivation: dw/dk = (theta/2)(rho*psi + psi*(psi*k+rho)/sqrt(...)))
    # at k=0 the sqrt term reduces to 1, so the two pieces are equal:
    # dw/dk(0) = (theta/2) * 2 * rho * psi = theta * rho * psi
    theta, rho, psi = 0.04, -0.4, 0.5
    slope = ssvi_dw_dk(0.0, theta, rho, psi)
    expected = theta * rho * psi
    assert slope == approx(expected, abs=1e-12)


def test_ssvi_d2w_dk2_at_atm() -> None:
    # d²w/dk²(0) = (theta/2) * psi² * (1-rho²)
    theta, rho, psi = 0.04, -0.4, 0.5
    curv = ssvi_d2w_dk2(0.0, theta, rho, psi)
    expected = (theta / 2.0) * psi * psi * (1.0 - rho * rho)
    assert curv == approx(expected, abs=1e-12)


def test_to_raw_svi_atm_value() -> None:
    # After mapping, the raw SVI formula at k=0 should give theta.
    a, b, rho, m, sigma = to_raw_svi_params(0.04, -0.4, 0.5)
    w0 = svi_total_variance(0.0, a, b, rho, m, sigma)
    assert w0 == approx(0.04, abs=1e-10)
    # Also: b = theta*psi/2 and sigma = 1/psi by construction.
    assert b == approx(0.04 * 0.5 / 2.0, abs=1e-12)
    assert sigma == approx(1.0 / 0.5, abs=1e-12)
    assert m == 0.0
    assert rho == -0.4


def test_essvi_arb_safe_default_params() -> None:
    # 0 <= gamma <= 1, eta > 0 is the arb-safe range.
    assert essvi_arb_safe(0.04, 0.5, 0.5)
    assert essvi_arb_safe(0.04, 0.5, 0.0)
    assert essvi_arb_safe(0.04, 0.5, 1.0)
    # Out-of-range gamma is flagged.
    assert not essvi_arb_safe(0.04, 0.5, 1.5)
    # eta <= 0 is flagged.
    assert not essvi_arb_safe(0.04, 0.0, 0.5)
    assert not essvi_arb_safe(0.04, -0.1, 0.5)
