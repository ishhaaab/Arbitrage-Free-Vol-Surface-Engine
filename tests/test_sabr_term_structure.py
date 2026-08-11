"""Tests for the SABR B-spline term-structure fitter."""

import numpy as np
import pytest
from pytest import approx

from arbfree_vol.sabr.model import SABRParams, sabr_total_variance
from arbfree_vol.sabr.calibration import calibrate_sabr
from arbfree_vol.sabr.term_structure import (
    EPS_FLOOR,
    fit_sabr_term_structure,
)

# Reproducible test parameters per slice (alpha slightly increasing)
_SLICE_PARAMS = [
    {"alpha": 0.20, "beta": 0.5, "rho": -0.3, "nu": 0.4},
    {"alpha": 0.22, "beta": 0.5, "rho": -0.25, "nu": 0.45},
    {"alpha": 0.25, "beta": 0.5, "rho": -0.2, "nu": 0.5},
]
_EXPIRIES = [0.25, 0.5, 1.0]
_FORWARD = 100.0
_K_GRID = np.linspace(-0.5, 0.5, 11)


def _make_synthetic_slices() -> list[tuple[float, float, list[tuple[float, float]]]]:
    """Build 3 synthetic slices from SABR total variance."""
    slices = []
    for T, p in zip(_EXPIRIES, _SLICE_PARAMS):
        points = [
            (float(k), sabr_total_variance(
                float(k), _FORWARD, T,
                p["alpha"], p["beta"], p["rho"], p["nu"],
            ))
            for k in _K_GRID
        ]
        slices.append((T, _FORWARD, points))
    return slices


# ---------------------------------------------------------------------------
# Test: valid params from term-structure fit
# ---------------------------------------------------------------------------

def test_fit_term_structure_produces_valid_params() -> None:
    """Term-structure fit on 3 synthetic slices returns valid SABRParams."""
    slices = _make_synthetic_slices()
    result = fit_sabr_term_structure(slices)

    assert len(result) == 3
    for i, p in enumerate(result):
        assert p.alpha > EPS_FLOOR, f"alpha[{i}]={p.alpha} <= EPS_FLOOR"
        assert p.nu > EPS_FLOOR, f"nu[{i}]={p.nu} <= EPS_FLOOR"
        assert -1.0 < p.rho < 1.0, f"rho[{i}]={p.rho} out of range"
        assert p.beta == 0.5


# ---------------------------------------------------------------------------
# Test: convex-hull guarantee — spline stays in-range between knots
# ---------------------------------------------------------------------------

def test_coefficient_reparametrization_keeps_curve_in_range() -> None:
    """The B-spline curve stays in-range between knots (convex-hull property).

    Uses a sparse-knot scenario where naive unbounded splines could overshoot.
    The fitted spline evaluated on a FINE grid between knots must yield
    alpha(t) > EPS_FLOOR, nu(t) > 0, rho(t) strictly in (-1, 1).
    """
    slices = _make_synthetic_slices()
    result, splines = fit_sabr_term_structure(slices, return_splines=True)

    assert len(result) == 3

    # Fine grid between the first and last expiry
    t_fine = np.linspace(_EXPIRIES[0], _EXPIRIES[-1], 200)

    alpha_fine = splines["alpha"](t_fine)
    nu_fine = splines["nu"](t_fine)
    rho_fine = splines["rho"](t_fine)

    for i, t in enumerate(t_fine):
        assert alpha_fine[i] > EPS_FLOOR, (
            f"alpha({t:.4f})={alpha_fine[i]:.8f} <= EPS_FLOOR"
        )
        assert nu_fine[i] > 0.0, (
            f"nu({t:.4f})={nu_fine[i]:.8f} <= 0"
        )
        assert -1.0 < rho_fine[i] < 1.0, (
            f"rho({t:.4f})={rho_fine[i]:.8f} out of (-1,1)"
        )


# ---------------------------------------------------------------------------
# Test: single-slice falls back to calibrate_sabr
# ---------------------------------------------------------------------------

def test_single_slice_falls_back_to_calibrate_sabr() -> None:
    """With a single slice, fit_sabr_term_structure delegates to calibrate_sabr."""
    T = 0.5
    p = _SLICE_PARAMS[0]
    points = [
        (float(k), sabr_total_variance(
            float(k), _FORWARD, T,
            p["alpha"], p["beta"], p["rho"], p["nu"],
        ))
        for k in _K_GRID
    ]
    slices = [(T, _FORWARD, points)]

    result = fit_sabr_term_structure(slices)

    # Also run calibrate_sabr directly for comparison
    direct = calibrate_sabr(points, forward=_FORWARD, expiry_time=T, beta_hint=0.5)

    assert len(result) == 1
    # The results should be approximately equal (both use calibrate_sabr)
    assert result[0].alpha == approx(direct.alpha, abs=1e-4)
    assert result[0].rho == approx(direct.rho, abs=1e-4)
    assert result[0].nu == approx(direct.nu, abs=1e-4)
    assert result[0].beta == direct.beta


# ---------------------------------------------------------------------------
# Test: return_splines=False is the default
# ---------------------------------------------------------------------------

def test_default_returns_just_list() -> None:
    """Default call returns list[SABRParams], not a tuple."""
    slices = _make_synthetic_slices()
    result = fit_sabr_term_structure(slices)
    assert isinstance(result, list)
    assert all(isinstance(p, SABRParams) for p in result)


def test_single_slice_calibration_failure_raises(monkeypatch) -> None:
    """When ``calibrate_sabr`` raises for a single-slice input,
    ``fit_sabr_term_structure`` propagates the RuntimeError instead of
    silently returning fabricated default params."""
    import arbfree_vol.sabr.term_structure as ts

    def _raise(*args, **kwargs):
        raise RuntimeError("calibration failed")

    monkeypatch.setattr(ts, "calibrate_sabr", _raise)

    slices = [(0.5, _FORWARD, [(0.0, 0.04)])]

    with pytest.raises(RuntimeError, match="calibration failed"):
        fit_sabr_term_structure(slices)


# ---------------------------------------------------------------------------
# Test: joint-fit nonconvergence falls back to per-slice calibrate_sabr
# ---------------------------------------------------------------------------

def test_joint_fit_nonconvergence_falls_back_to_per_slice(monkeypatch, caplog) -> None:
    """When the joint B-spline least-squares fit reports
    ``success == False``, ``fit_sabr_term_structure`` must log the
    documented warning and return the per-slice ``calibrate_sabr``
    results — the returned parameters must equal (within tolerance) an
    independent direct ``calibrate_sabr`` call on the same slices.

    The fallback is forced deterministically: the joint-fit call is
    monkeypatched to return a non-converged result object, so the branch
    does not depend on optimizer budgets.  This pins the documented
    contract in the docstring: there is NO marker distinguishing a
    fallback result from a converged joint fit — value-equals-per-slice
    is the fallback contract."""
    import logging
    import arbfree_vol.sabr.term_structure as ts

    slices = _make_synthetic_slices()

    class _NonConverged:
        success = False
        message = "simulated joint-fit nonconvergence"

    monkeypatch.setattr(ts, "least_squares", lambda *a, **k: _NonConverged())

    with caplog.at_level(logging.WARNING, logger="arbfree_vol.sabr.term_structure"):
        result = fit_sabr_term_structure(slices)

    # Fallback contract: per-slice params equal a direct calibrate_sabr
    # call on the same slices (within solver determinism).
    assert len(result) == len(slices)
    for i, (T, F, pts) in enumerate(slices):
        direct = calibrate_sabr(pts, forward=F, expiry_time=T, beta_hint=0.5)
        assert result[i].alpha == approx(direct.alpha, abs=1e-10)
        assert result[i].rho == approx(direct.rho, abs=1e-10)
        assert result[i].nu == approx(direct.nu, abs=1e-10)
        assert result[i].beta == direct.beta

    # The documented warning is emitted.
    assert "falling back to per-slice calibrate_sabr" in caplog.text, (
        f"expected the fallback warning, got: {caplog.text}"
    )
