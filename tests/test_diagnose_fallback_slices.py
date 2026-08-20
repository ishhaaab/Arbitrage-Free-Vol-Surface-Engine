"""Tests for the eSSVI fallback-slice diagnostics library.

Covers ``arbfree_vol.ssvi.diagnostics`` (promoted from the former
``scripts/diagnose_fallback_slices.py`` research script):

- predecessor-selection edge cases that the diagnostic loop relies on
  (unsorted fitted-slice input and repeated parameter-object identity
  across maturities);
- the synthetic W7 fixture (vol hump + rho flip) that deterministically
  triggers 3 fallback slices on the current fitter at
  T = 0.427 / 0.75 / 1.00;
- slice analytics and constraint-violation reporting;
- the hard-constrained fit attempts (default seed, warm-start, random
  restarts) against the fixture's real predecessor chain;
- the H&M neighbour check;
- golden-parity pins for ``run_diagnostics`` end-to-end.

The end-to-end and convergence-table pins rely on scipy optimizer
determinism (trust-constr with an SLSQP retry); the fixture was
verified deterministic on scipy 1.17.1.
"""

import numpy as np
import pytest
from types import SimpleNamespace

from arbfree_vol.ssvi.calibration import fit_ssvi_slice
from arbfree_vol.ssvi.model import SSVIParams, ssvi_w
from arbfree_vol.ssvi.term_structure import fit_ssvi_surface_sequential
from arbfree_vol.ssvi.diagnostics import (
    _build_synthetic_data,
    _build_violation_info,
    _check_hm_neighbors,
    _find_predecessor,
    _fit_unconstrained,
    _print_predecessor,
    _print_summary_and_interpretation,
    check_unconstrained_satisfies_hm,
    compute_slice_analytics,
    extract_slice_data,
    run_diagnostics,
    try_hard_constrained,
    try_random_restarts,
    try_warm_start,
)


# ── Shared fixture ───────────────────────────────────────────────────

_W7_EXPIRIES = [0.03, 0.085, 0.13, 0.18, 0.258, 0.33, 0.427, 0.5, 0.75, 1.0, 1.5, 2.0]
_W7_FALLBACKS = [0.427, 0.75, 1.0]


@pytest.fixture(scope="module")
def synthetic_fit():
    """Sequential fit of the W7 synthetic fixture (PRECEDENCE-dependent!).

    The fallback failures only occur with the real predecessor chain:
    every fixture slice converges hard-constrained with ``prev=None``.
    """
    surface, _ = _build_synthetic_data()
    slices_data = extract_slice_data(surface)
    result = fit_ssvi_surface_sequential(slices_data)
    return by_T(slices_data), result


def by_T(slices_data):
    return {T: pts for T, pts in slices_data}


# ── Predecessor selection ────────────────────────────────────────────


def _p(theta: float) -> SSVIParams:
    return SSVIParams(theta=theta, rho=-0.3, psi=0.5)


def test_find_predecessor_returns_last_below_T() -> None:
    fitted = [(0.1, _p(1.0)), (0.2, _p(2.0)), (0.3, _p(3.0))]
    params, T = _find_predecessor(fitted, 0.25)
    assert T == 0.2
    assert params is not None
    assert params.theta == 2.0


def test_find_predecessor_unsorted_input() -> None:
    # Input order must not matter: selection is by maturity.
    fitted = [(0.3, _p(3.0)), (0.1, _p(1.0)), (0.2, _p(2.0))]
    params, T = _find_predecessor(fitted, 0.25)
    assert T == 0.2
    assert params is not None
    assert params.theta == 2.0


def test_find_predecessor_returns_none_when_no_predecessor() -> None:
    fitted = [(0.1, _p(1.0)), (0.2, _p(2.0))]
    params, T = _find_predecessor(fitted, 0.05)
    assert params is None
    assert T is None


def test_find_predecessor_duplicate_identity_returns_matching_T() -> None:
    # A single params object shared by two maturities: the returned T must
    # be the maturity actually selected (the last one strictly below T),
    # not a re-derived match from an unsorted scan.
    shared = _p(7.0)
    fitted = [(0.1, shared), (0.2, shared), (0.3, shared)]
    params, T = _find_predecessor(fitted, 0.25)
    assert params is shared
    assert T == 0.2


# ── Synthetic fixture structure ──────────────────────────────────────


def test_build_synthetic_data_structure() -> None:
    surface, rejected = _build_synthetic_data()

    assert surface.spot == pytest.approx(550.0)
    assert rejected == []
    assert len(surface.slices) == 12

    Ts = [sl.expiry_time for sl in surface.slices]
    assert Ts == sorted(Ts)
    assert Ts == _W7_EXPIRIES

    # Every slice has enough quotes to fit, all economically positive.
    total = 0
    for sl in surface.slices:
        assert len(sl.quotes) >= 5
        for q in sl.quotes:
            assert q.price > 0.01
        total += len(sl.quotes)
    assert total == 499  # determinism pin (21 strikes x 2 types, cheap OTM dropped)


def test_extract_slice_data_preserves_12_sorted_slices() -> None:
    surface, _ = _build_synthetic_data()
    slices_data = extract_slice_data(surface)

    assert len(slices_data) == 12
    Ts = [T for T, _ in slices_data]
    assert Ts == sorted(Ts)
    for T, pts in slices_data:
        assert len(pts) == 21
        ks = [k for k, _ in pts]
        assert ks == sorted(ks)


# ── Slice analytics ──────────────────────────────────────────────────


def test_compute_slice_analytics_known_smile() -> None:
    T = 0.5
    theta, psi = 0.02, 0.6
    ks = np.linspace(-1.2, 1.2, 25)  # grid includes k=0 exactly
    pts_neg = [(float(k), ssvi_w(float(k), theta, -0.3, psi)) for k in ks]
    pts_flat = [(float(k), ssvi_w(float(k), theta, 0.0, psi)) for k in ks]
    pts_pos = [(float(k), ssvi_w(float(k), theta, 0.3, psi)) for k in ks]

    a = compute_slice_analytics(T, pts_neg)

    assert a["n_points"] == 25
    assert a["atm_w"] == pytest.approx(theta)  # w(0) == theta
    assert a["atm_vol"] == pytest.approx(np.sqrt(theta / T))
    assert a["k_range"][0] == pytest.approx(-1.2)
    assert a["k_range"][1] == pytest.approx(1.2)

    # Skew properties that are independent of the measure's sign convention:
    # rho=0 -> symmetric smile -> zero skew; mirrored rho -> mirrored skew;
    # tails are more skewed than 25-delta.
    a_flat = compute_slice_analytics(T, pts_flat)
    a_pos = compute_slice_analytics(T, pts_pos)
    assert a_flat["skew_25d"] == pytest.approx(0.0, abs=1e-12)
    assert a_pos["skew_25d"] == pytest.approx(-a["skew_25d"])
    assert a_pos["skew_10d"] == pytest.approx(-a["skew_10d"])
    assert abs(a["skew_10d"]) > abs(a["skew_25d"]) > 0


def test_compute_slice_analytics_interp_clamps_tails() -> None:
    # Interpolation with a 25-delta log-moneyness beyond the grid range
    # clamps at the edges and stays finite.
    T = 0.5
    ks = np.linspace(-0.5, 0.5, 21)
    pts = [(float(k), ssvi_w(float(k), 0.02, -0.3, 0.6)) for k in ks]
    a = compute_slice_analytics(T, pts)

    assert np.isfinite(a["skew_25d"])
    assert np.isfinite(a["skew_10d"])


# ── Constraint-violation info ────────────────────────────────────────


def test_build_violation_info_no_prev() -> None:
    params = SSVIParams(theta=0.05, rho=-0.3, psi=0.5)
    info = _build_violation_info(params, None)

    assert info["theta"] == 0.05
    assert info["rho"] == -0.3
    assert info["psi"] == 0.5
    assert info["chi"] == pytest.approx(0.025)
    assert isinstance(info["bf_min_residual"], float)
    assert "theta_delta" not in info  # calendar keys only with a predecessor


def test_build_violation_info_with_prev() -> None:
    prev = SSVIParams(theta=0.04, rho=-0.2, psi=0.5)   # chi = 0.02
    params = SSVIParams(theta=0.06, rho=-0.3, psi=0.5)  # chi = 0.03
    info = _build_violation_info(params, prev)

    assert info["theta_delta"] == pytest.approx(0.02)
    assert info["chi_delta"] == pytest.approx(0.01)
    # ratio = (rho*chi - prev_rho*prev_chi) / (chi - prev_chi)
    #       = (-0.009 - (-0.004)) / 0.01 = -0.5
    assert info["ratio"] == pytest.approx(-0.5)
    assert info["ratio_ok"] is True


# ── Hard-constrained fit attempts (real predecessor chain) ───────────


def test_try_hard_constrained_converges_on_clean_slice(synthetic_fit) -> None:
    pts_by_T, result = synthetic_fit
    pts = pts_by_T[0.03]

    # 0.03 is the first slice: no predecessor anywhere in the fit.
    prev, prev_T = _find_predecessor(result.fitted_slices, 0.03)
    assert prev is None and prev_T is None

    r = try_hard_constrained(pts, prev)
    assert r["label"] == "default"
    assert r["converged"] is True
    assert r["params"] is not None
    assert "bf_min_residual" in r["violations"]


def test_try_hard_constrained_fails_on_fallback_slice(synthetic_fit) -> None:
    pts_by_T, result = synthetic_fit
    pts = pts_by_T[0.427]
    prev, prev_T = _find_predecessor(result.fitted_slices, 0.427)
    assert prev_T == pytest.approx(0.33)  # last HARD fit before the fallback

    r = try_hard_constrained(pts, prev)
    assert r["converged"] is False
    assert r["params"] is None
    assert "failed after retry" in r["error"]


def test_try_warm_start_fixes_fixable_slice(synthetic_fit) -> None:
    pts_by_T, result = synthetic_fit
    pts = pts_by_T[0.427]
    prev, _ = _find_predecessor(result.fitted_slices, 0.427)
    unc = fit_ssvi_slice(pts)

    r = try_warm_start(pts, prev, unc)
    assert r["label"] == "warm-start"
    assert r["converged"] is True
    assert r["params"] is not None
    assert r["optimizer_status"] == 0
    assert isinstance(r["final_objective"], float)
    assert len(r["x0"]) == 3
    assert r["start"].theta > 0.0
    assert "bf_min_residual" in r["violations"]


def test_try_warm_start_fails_on_infeasible_slice(synthetic_fit) -> None:
    pts_by_T, result = synthetic_fit
    pts = pts_by_T[0.75]
    prev, prev_T = _find_predecessor(result.fitted_slices, 0.75)
    assert prev_T == pytest.approx(0.5)  # the rho-flip slice
    unc = fit_ssvi_slice(pts)

    r = try_warm_start(pts, prev, unc)
    assert r["converged"] is False
    assert r["params"] is None
    assert set(r.keys()) == {
        "label", "start", "x0", "converged", "params",
        "error", "final_objective", "optimizer_status", "optimizer_message",
    }
    assert r["optimizer_status"] > 0


def test_try_random_restarts_contract(synthetic_fit) -> None:
    pts_by_T, _ = synthetic_fit
    pts = pts_by_T[0.03]
    rs = try_random_restarts(pts, None, n_restarts=2)

    assert len(rs) == 2
    for i, r in enumerate(rs):
        assert r["label"] == f"restart-{i}"
        assert r["start"].theta > 0.0
        assert len(r["x0"]) == 3
        assert isinstance(r["converged"], bool)
        assert isinstance(r["optimizer_status"], int)
    # Clean slice: every independent restart converges.
    assert sum(1 for r in rs if r["converged"]) == 2


def test_unconstrained_fit_reports_failure(monkeypatch) -> None:
    pts = [(0.0, 0.02), (0.1, 0.021)]

    def _boom(_pts):
        raise RuntimeError("fit exploded")

    monkeypatch.setattr("arbfree_vol.ssvi.diagnostics.fit_ssvi_slice", _boom)
    params, rmse = _fit_unconstrained(pts)
    assert params is None
    assert np.isnan(rmse)


# ── H&M neighbour check ──────────────────────────────────────────────


def _hm_triple():
    p1 = SSVIParams(theta=0.04, rho=-0.3, psi=0.5)   # chi 0.02
    p2 = SSVIParams(theta=0.08, rho=-0.2, psi=0.6)   # chi 0.048
    p3 = SSVIParams(theta=0.14, rho=-0.1, psi=0.65)  # chi 0.091
    return [(0.1, p1), (0.25, p2), (0.5, p3)], p1, p2, p3


def test_check_unconstrained_satisfies_hm_holds_middle() -> None:
    fitted, _, p2, _ = _hm_triple()
    chk = check_unconstrained_satisfies_hm(fitted, 0.25, p2)

    assert chk["satisfies_with_prev"] is True
    assert chk["satisfies_with_next"] is True
    assert chk["satisfies_both"] is True
    assert chk["details"]["prev_pair"]["theta_delta"] == pytest.approx(0.04)
    assert chk["details"]["next_pair"]["theta_delta"] == pytest.approx(0.06)


def test_check_unconstrained_satisfies_hm_theta_drop() -> None:
    # A theta drop below the predecessor violates H&M condition (a).
    fitted, _, _, _ = _hm_triple()
    p2_drop = SSVIParams(theta=0.03, rho=-0.2, psi=0.6)
    chk = check_unconstrained_satisfies_hm(fitted, 0.25, p2_drop)

    assert chk["satisfies_with_prev"] is False
    assert chk["satisfies_with_next"] is True
    assert chk["satisfies_both"] is False


def test_check_unconstrained_satisfies_hm_first_slice() -> None:
    fitted, p1, _, _ = _hm_triple()
    chk = check_unconstrained_satisfies_hm(fitted, 0.1, p1)

    # No predecessor: the with-prev flag stays at its default True.
    assert chk["satisfies_with_prev"] is True
    assert chk["satisfies_with_next"] is True
    assert chk["satisfies_both"] is True
    assert "prev_pair" not in chk["details"]
    assert "next_pair" in chk["details"]


def test_check_unconstrained_satisfies_hm_missing_T() -> None:
    fitted, _, _, _ = _hm_triple()
    chk = check_unconstrained_satisfies_hm(fitted, 0.77, fitted[0][1])

    assert chk["satisfies_both"] is False
    assert chk["details"] == {}


def test_check_hm_neighbors_none_params() -> None:
    fitted, _, _, _ = _hm_triple()
    chk = _check_hm_neighbors(None, fitted, 0.25)
    assert chk == {"satisfies_both": False}


# ── Report helpers ───────────────────────────────────────────────────


def _row(T, default, warm, restart, hm):
    return {
        "T": T,
        "unc_rmse": 1e-10,
        "default_converged": default,
        "warm_start_converged": warm,
        "restart_converged": restart,
        "unc_satisfies_hm": hm,
    }


def test_print_summary_and_interpretation_outcome_a(capsys) -> None:
    rows = [
        _row(0.427, False, True, 5, False),
        _row(0.75, False, True, 4, True),
    ]
    _print_summary_and_interpretation(rows)
    out = capsys.readouterr().out

    assert "OUTCOME A" in out
    assert "warm-start" in out.lower()
    assert "Total fallback slices: 2" in out
    assert "Warm-start fixes: 2" in out
    assert "5/5" in out and "4/5" in out


def test_print_summary_and_interpretation_outcome_c(capsys) -> None:
    rows = [
        _row(0.427, False, False, 0, False),
        _row(0.75, True, True, 5, False),
    ]
    _print_summary_and_interpretation(rows)
    out = capsys.readouterr().out

    assert "OUTCOME C" in out
    assert "Warm-start fixes: 0" in out


def test_print_predecessor_none_branch(capsys) -> None:
    _print_predecessor(None, None)
    out = capsys.readouterr().out
    assert "No predecessor" in out


# ── run_diagnostics pipeline ─────────────────────────────────────────


def test_run_diagnostics_no_fallback_short_circuit(monkeypatch, capsys) -> None:
    import arbfree_vol.ssvi.diagnostics as diag

    monkeypatch.setattr(diag, "fetch_spy_data", diag._build_synthetic_data)
    monkeypatch.setattr(
        diag,
        "fit_ssvi_surface_sequential",
        lambda _sd: SimpleNamespace(
            fitted_slices=[],
            fallback_slices=[],
            failed_slices=[],
        ),
    )

    assert run_diagnostics() is None
    assert "No fallback slices found. Nothing to diagnose." in capsys.readouterr().out


def test_run_diagnostics_missing_slice_data_reports_error(
    monkeypatch, capsys
) -> None:
    import arbfree_vol.ssvi.diagnostics as diag

    monkeypatch.setattr(diag, "fetch_spy_data", diag._build_synthetic_data)
    monkeypatch.setattr(
        diag,
        "fit_ssvi_surface_sequential",
        lambda _sd: SimpleNamespace(
            fitted_slices=[],
            fallback_slices=[0.999],  # T with no extracted data
            failed_slices=[],
        ),
    )

    rows = run_diagnostics()
    out = capsys.readouterr().out
    assert rows == []
    assert "ERROR: No data found for this T" in out
    assert "OUTCOME C" in out  # empty interpretation falls through


@pytest.mark.slow
def test_run_diagnostics_end_to_end_golden_parity(monkeypatch, capsys) -> None:
    """End-to-end synthetic run — golden parity with the pre-promotion script.

    The W7 fixture deterministically produces 3 fallback slices at
    T = 0.427 / 0.75 / 1.00 (verified on scipy 1.17.1); this test pins the
    exact convergence matrix the diagnostic reports for them.
    """
    import arbfree_vol.ssvi.diagnostics as diag

    monkeypatch.setattr(diag, "fetch_spy_data", diag._build_synthetic_data)
    rows = run_diagnostics()

    assert rows is not None
    assert len(rows) == 3

    expected = {
        0.427: dict(default=False, warm=True, restart=5, hm=False),
        0.75: dict(default=False, warm=False, restart=0, hm=False),
        1.0: dict(default=True, warm=True, restart=5, hm=False),
    }
    for row in rows:
        T = row["T"]
        assert T in expected, f"unexpected fallback T={T}"
        assert row["unc_rmse"] < 1e-6  # fixture is exactly SSVI-representable
        assert row["default_converged"] is expected[T]["default"]
        assert row["warm_start_converged"] is expected[T]["warm"]
        assert row["restart_converged"] == expected[T]["restart"]
        assert row["unc_satisfies_hm"] is expected[T]["hm"]

    out = capsys.readouterr().out
    assert "OUTCOME B" in out
    assert "SUMMARY TABLE" in out
    assert "Warm-start fixes: 1" in out
    assert "Fundamental infeasibility: 2" in out
    for T in _W7_FALLBACKS:
        assert f"T={T:.4f}" in out