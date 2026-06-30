"""Tests for SVI formula and calibration."""

import numpy as np
import pytest
from pytest import approx

from arbfree_vol.svi.calibration import calibrate
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
