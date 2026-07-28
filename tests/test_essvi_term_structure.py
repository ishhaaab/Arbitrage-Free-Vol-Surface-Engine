"""Tests for the eSSVI sequential term-structure fitter.

Verifies the Hendriks & Martini (2019) Prop 3.1 no-calendar-spread
conditions hold after calibration.

Reference: Hendriks & Martini (2019), "The Extended SSVI Volatility
Surface", J. Comput. Finance 22(5), 25-39, Prop 3.1.
"""

import numpy as np
import pytest
from math import sqrt

from arbfree_vol.ssvi.model import SSVIParams, ssvi_w, to_raw_svi_params
from arbfree_vol.ssvi.term_structure import (
    fit_ssvi_surface_sequential,
    verify_hm_condition,
)
from arbfree_vol.arbitrage.svi_detect import detect_svi_surface


# ── Synthetic ground truth ──────────────────────────────────────────
# Three slices with theta increasing and chi = theta*psi increasing.
_TRUTH = [
    dict(theta=0.04, rho=-0.3, psi=0.5),   # T=0.25
    dict(theta=0.08, rho=-0.2, psi=0.6),   # T=0.50
    dict(theta=0.14, rho=-0.1, psi=0.65),  # T=1.00
]

_EXPIRIES = [0.25, 0.50, 1.00]

_KS = np.linspace(-1.0, 1.0, 9).tolist()


def _make_slices_data():
    """Build synthetic (expiry, [(k, w)]) data from ground-truth SSVI."""
    slices = []
    for T, truth in zip(_EXPIRIES, _TRUTH):
        pts = [
            (float(k), ssvi_w(float(k), truth["theta"], truth["rho"], truth["psi"]))
            for k in _KS
        ]
        slices.append((T, pts))
    return slices


# ── Test 1 ──────────────────────────────────────────────────────────
def test_theta_and_chi_non_decreasing() -> None:
    """After calibration, theta_i and chi_i=theta_i*psi_i must be
    non-decreasing across slices (Hendriks & Martini 2019, Prop 3.1,
    conditions (a) and (b))."""
    slices_data = _make_slices_data()
    result = fit_ssvi_surface_sequential(slices_data)
    params = [p for _, p in result]

    assert len(params) == 3

    for i in range(len(params) - 1):
        assert params[i + 1].theta >= params[i].theta - 1e-8, (
            f"theta not non-decreasing at slice {i}: "
            f"{params[i].theta} > {params[i + 1].theta}"
        )
        chi_i = params[i].theta * params[i].psi
        chi_next = params[i + 1].theta * params[i + 1].psi
        assert chi_next >= chi_i - 1e-8, (
            f"chi not non-decreasing at slice {i}: "
            f"{chi_i} > {chi_next}"
        )


# ── Test 2 ──────────────────────────────────────────────────────────
def test_pairwise_inequality_holds() -> None:
    """For each adjacent pair the discrete Prop 3.1 inequality must hold:

    | rho_{i+1}*chi_{i+1} - rho_i*chi_i | / (chi_{i+1} - chi_i) <= 1

    (Hendriks & Martini 2019, Prop 3.1, condition (c)).
    """
    slices_data = _make_slices_data()
    result = fit_ssvi_surface_sequential(slices_data)
    params = [p for _, p in result]

    chis = [p.theta * p.psi for p in params]
    tol = 1e-8

    for i in range(len(params) - 1):
        denom = chis[i + 1] - chis[i]
        denom = max(denom, tol)
        num = params[i + 1].rho * chis[i + 1] - params[i].rho * chis[i]
        ratio = abs(num) / denom
        assert ratio <= 1.0 + tol, (
            f"pairwise inequality violated at pair ({i}, {i+1}): "
            f"|ratio| = {ratio:.10f} > 1"
        )


# ── Test 3 ──────────────────────────────────────────────────────────
def test_grid_calendar_detector_reports_zero() -> None:
    """The grid-based detect_svi_surface must report zero violations on
    a calibrated eSSVI surface (redundant regression check)."""
    slices_data = _make_slices_data()
    result = fit_ssvi_surface_sequential(slices_data)
    params = [p for _, p in result]

    svi_pairs: list[tuple[float, object]] = []
    for T, p in zip(_EXPIRIES, params):
        a, b, rho, m, sigma = to_raw_svi_params(p.theta, p.rho, p.psi)
        from arbfree_vol.svi.model import SVIParams
        svi_pairs.append((T, SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)))

    report = detect_svi_surface(svi_pairs)
    assert report.is_arbitrage_free, (
        f"detect_svi_surface found {len(report.violations)} violation(s) "
        f"on a calibrated eSSVI surface: "
        f"{[v.detail for v in report.violations]}"
    )


# ── Test 4 ──────────────────────────────────────────────────────────
def test_near_equal_chi_no_divide_by_zero() -> None:
    """When chi_2 is very close to chi_1 the fitter must not produce
    NaN/inf and verify_hm_condition must still return True.

    We feed a near-flat surface (identical w across two expiries) so
    the optimizer is pushed towards chi_2 ≈ chi_1.  The eps_chi floor
    prevents division by zero in the pairwise inequality constraint.
    """
    # Two slices with very similar data => chi should stay close
    flat_w = 0.04  # constant total variance
    pts = [(float(k), flat_w) for k in np.linspace(-1.0, 1.0, 9)]
    slices_data = [
        (0.25, pts),
        (0.50, pts),  # identical points => forces chi close
    ]

    params = fit_ssvi_surface_sequential(slices_data)
    assert len(params) == 2

    # All params must be finite
    for _T, p in params:
        assert np.isfinite(p.theta), f"theta is not finite: {p.theta}"
        assert np.isfinite(p.rho), f"rho is not finite: {p.rho}"
        assert np.isfinite(p.psi), f"psi is not finite: {p.psi}"

    # verify_hm_condition must pass (the eps_chi floor handles denom → 0)
    assert verify_hm_condition([p for _, p in params]), (
        "verify_hm_condition returned False on near-equal-chi slices"
    )


# ── Test 5: fallback on infeasible slice ────────────────────────────
def test_sequential_fit_falls_back_on_infeasible_slice(monkeypatch) -> None:
    """When a slice's data makes the H&M constraints infeasible against
    the predecessor, the fitter must fall back to the unconstrained
    per-slice fit rather than raising.

    We build a 3-slice surface with normal data and monkeypatch
    ``_fit_slice`` to raise ``RuntimeError`` on the second call (the
    middle slice), simulating an infeasible H&M constraint.  The
    sequential fitter must fall back to ``fit_ssvi_slice``.

    The result must contain all 3 slices.  The fallback slice is NOT
    arb-free by construction, so verify_hm_condition may report a
    violation — that's expected and honest.
    """
    import arbfree_vol.ssvi.term_structure as ts

    truth1 = dict(theta=0.04, rho=-0.3, psi=0.5)
    truth2 = dict(theta=0.08, rho=-0.2, psi=0.6)
    truth3 = dict(theta=0.14, rho=-0.1, psi=0.65)
    ks = np.linspace(-1.0, 1.0, 9).tolist()

    def _pts(truth):
        return [
            (float(k), ssvi_w(float(k), truth["theta"], truth["rho"], truth["psi"]))
            for k in ks
        ]

    slices_data = [
        (0.25, _pts(truth1)),
        (0.50, _pts(truth2)),
        (1.00, _pts(truth3)),
    ]

    # Monkeypatch _fit_slice to fail on the 2nd call
    call_count = {"n": 0}
    _real_fit_slice = ts._fit_slice

    def _failing_fit_slice(points, prev=None, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated infeasible H&M constraints")
        return _real_fit_slice(points, prev=prev, **kwargs)

    monkeypatch.setattr(ts, "_fit_slice", _failing_fit_slice)

    result = fit_ssvi_surface_sequential(slices_data)

    # All 3 slices must be present (hard or fallback)
    assert len(result) == 3, (
        f"expected 3 entries, got {len(result)}"
    )

    # The middle slice should have fallen back to unconstrained
    print(f"  result expiries: {[T for T, _ in result]}")
    for i, (T, p) in enumerate(result):
        print(f"  slice {i} (T={T}): theta={p.theta:.6f}, rho={p.rho:.4f}, psi={p.psi:.4f}")

    # verify_hm_condition may be False because of the fallback
    params_only = [p for _, p in result]
    hm_ok = verify_hm_condition(params_only)
    print(f"  verify_hm_condition: {hm_ok}")
    # We don't assert True or False — just that the function returns
    # without raising.
