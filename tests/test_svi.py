"""Tests for SVI formula and calibration."""

import numpy as np
import pytest
from pytest import approx

from arbfree_vol.svi.calibration import calibrate, calibrate_constrained
from arbfree_vol.svi.model import SVIParams, svi_total_variance

# A known-good, well-identifiable SVI smile.
TRUE = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.05, sigma=0.15)


def _points_from(params: SVIParams, ks: np.ndarray) -> list[tuple[float, float]]:
    return [
        (float(k), svi_total_variance(float(k), params.a, params.b, params.rho, params.m, params.sigma))
        for k in ks
    ]


def test_svi_atm_value() -> None:
    # At k = m the sqrt term collapses to sigma, so w = a + b*sigma.
    w = svi_total_variance(TRUE.m, TRUE.a, TRUE.b, TRUE.rho, TRUE.m, TRUE.sigma)
    assert w == approx(TRUE.a + TRUE.b * TRUE.sigma, abs=1e-12)


def test_calibrate_recovers_known_params() -> None:
    points = _points_from(TRUE, np.linspace(-0.4, 0.4, 9))

    fit = calibrate(points)

    assert fit.a == approx(TRUE.a, abs=1e-3)
    assert fit.b == approx(TRUE.b, abs=1e-3)
    assert fit.rho == approx(TRUE.rho, abs=1e-3)
    assert fit.m == approx(TRUE.m, abs=1e-3)
    assert fit.sigma == approx(TRUE.sigma, abs=1e-3)


def test_calibrated_curve_fits_the_cloud() -> None:
    ks = np.linspace(-0.4, 0.4, 9)
    points = _points_from(TRUE, ks)

    fit = calibrate(points)

    # The fitted curve should reproduce every data point essentially exactly.
    for k, w in points:
        w_fit = svi_total_variance(k, fit.a, fit.b, fit.rho, fit.m, fit.sigma)
        assert w_fit == approx(w, abs=1e-6)


def test_calibrate_raises_with_too_few_points() -> None:
    with pytest.raises(ValueError):
        calibrate([(-0.1, 0.05), (0.0, 0.04), (0.1, 0.05)])


# ---------------------------------------------------------------------------
# Constrained calibration tests
# ---------------------------------------------------------------------------

TRUE_FLAT = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.0, sigma=0.15)


def test_calibrate_constrained_recovers_known_params() -> None:
    """calibrate_constrained should recover known params (looser tolerance
    due to penalty terms nudging the optimizer)."""
    points = _points_from(TRUE_FLAT, np.linspace(-0.4, 0.4, 9))
    fit = calibrate_constrained(points)

    assert fit.a == approx(TRUE_FLAT.a, abs=1e-2)
    assert fit.b == approx(TRUE_FLAT.b, abs=1e-2)
    assert fit.rho == approx(TRUE_FLAT.rho, abs=1e-2)
    assert fit.m == approx(TRUE_FLAT.m, abs=1e-2)
    assert fit.sigma == approx(TRUE_FLAT.sigma, abs=1e-2)


def test_calibrate_constrained_clean_input_is_arb_free() -> None:
    """Clean SVI data should produce an arb-free fit under constrained
    calibration."""
    from arbfree_vol.arbitrage.svi_detect import detect_svi

    points = _points_from(TRUE_FLAT, np.linspace(-0.4, 0.4, 9))
    fit = calibrate_constrained(points)
    report = detect_svi(fit)
    assert report.is_arbitrage_free


# Noise fixture for the adversarial test below.
# TRUE2 parameters: a=0.001, b=0.8, rho=-0.7, m=0.0, sigma=0.05.
# These represent an aggressive SVI smile with steep wings and strong
# negative skew.  Adding multiplicative noise w * (1 + 0.20 * sin(3*k))
# creates an asymmetric wavy pattern that tricks the unconstrained
# least-squares fit into producing a butterfly-violating curve, while
# the constrained fit (which penalises g(k) < 0) stays arb-free.
TRUE2 = SVIParams(a=0.001, b=0.8, rho=-0.7, m=0.0, sigma=0.05)
_NOISE_AMP = 0.20
_NOISE_FREQ = 3.0


def _noisy_points_from(params: SVIParams, ks: np.ndarray) -> list[tuple[float, float]]:
    return [
        (
            float(k),
            svi_total_variance(float(k), params.a, params.b, params.rho,
                                params.m, params.sigma)
            * (1.0 + _NOISE_AMP * np.sin(_NOISE_FREQ * float(k))),
        )
        for k in ks
    ]


def test_calibrate_constrained_better_than_unconstrained_on_noisy_input() -> None:
    """On a noisy aggressive SVI smile, unconstrained ``calibrate()``
    produces a butterfly-violating curve while ``calibrate_constrained()``
    does not."""
    from arbfree_vol.arbitrage.svi_detect import detect_svi

    ks = np.linspace(-0.5, 0.5, 15)
    noisy_points = _noisy_points_from(TRUE2, ks)

    fit_unconstrained = calibrate(noisy_points)
    fit_constrained = calibrate_constrained(noisy_points)

    unconstrained_report = detect_svi(fit_unconstrained)
    constrained_report = detect_svi(fit_constrained)

    assert not unconstrained_report.is_arbitrage_free, (
        "Unconstrained fit must produce a butterfly violation "
        "to demonstrate constrained > unconstrained"
    )
    assert constrained_report.is_arbitrage_free, (
        "Constrained fit must remain arb-free on the same noisy data"
    )


def test_calibrate_constrained_raises_with_too_few_points() -> None:
    with pytest.raises(ValueError):
        calibrate_constrained([(-0.1, 0.05), (0.0, 0.04), (0.1, 0.05)])
