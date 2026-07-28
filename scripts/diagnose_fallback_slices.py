"""Diagnostic script: investigate why 3 SPY eSSVI slices take the fallback path.

The 3 fallback T values: 0.0849, 0.2575, 0.4274.
The fallback works (unconstrained fit succeeds) but the hard-constrained
H&M Prop 3.1 fit fails on these slices.

This script:
  1. Fetches live SPY data (yfinance)
  2. Runs fit_ssvi_surface_sequential and identifies fallback slices
  3. For each fallback slice, runs detailed diagnostics:
     a. Slice data (T, ATM vol, 25-delta skew, 10-delta skew)
     b. Unconstrained fit params (theta, psi, rho)
     c. Hard-constrained fit attempts with:
        - Default seed (current behavior)
        - Warm-start from unconstrained fit params
        - 5 random restarts
     d. Constraint violation at each solution
     e. Whether unconstrained params already satisfy H&M with neighbors
  4. Prints a summary table

Usage:
    python scripts/diagnose_fallback_slices.py
"""

import sys
import os
import logging
from math import log, sqrt
from pathlib import Path

import numpy as np

# Ensure the project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arbfree_vol.ssvi.model import SSVIParams, ssvi_w
from arbfree_vol.ssvi.calibration import fit_ssvi_slice
from arbfree_vol.ssvi.term_structure import (
    fit_ssvi_surface_sequential,
    verify_hm_condition,
    _fit_slice,
    _butterfly_constraints,
)
from arbfree_vol.variance import slice_total_variance
from arbfree_vol.models.surface import VolSurface, ExpirySlice
from arbfree_vol.models.option import OptionType
from arbfree_vol.ingestion.cleaning import clean_quotes
from arbfree_vol.repair.fwd_curve import estimate_forward_curve, populate_per_slice_r

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


# ── Data fetching ────────────────────────────────────────────────────


def fetch_spy_data() -> tuple[VolSurface, list]:
    """Fetch SPY option data via yfinance, same approach as the demo."""
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed. Using synthetic fallback data.")
        return _build_synthetic_data()

    from arbfree_vol.ingestion.yfinance import fetch_chain
    try:
        surface, rejected = fetch_chain("SPY", max_expiries=20, min_T_years=7.0 / 365.0)
        return surface, rejected
    except Exception as e:
        print(f"yfinance fetch failed ({e}). Using synthetic fallback data.")
        return _build_synthetic_data()


def _build_synthetic_data() -> tuple[VolSurface, list]:
    """Build synthetic SPY-like data that triggers the same fallback.

    We construct a term structure with a vol hump at 0.25-0.5y and skew
    inversion, which is the typical trigger for H&M constraint failure.
    """
    np.random.seed(42)
    spot = 550.0
    r, q = 0.05, 0.013

    strikes_base = [spot * x for x in [
        0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99,
        1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.08, 1.10, 1.12, 1.15,
    ]]

    # Term structure with a vol hump: vol rises then falls
    expiry_configs = [
        (0.03,  0.22, -0.45, 0.0),   # very short: lower vol
        (0.085, 0.28, -0.50, 0.0),   # near the hump
        (0.13,  0.30, -0.48, 0.0),
        (0.18,  0.31, -0.46, 0.0),
        (0.258, 0.32, -0.44, 0.0),   # hump peak
        (0.33,  0.31, -0.42, 0.0),   # declining
        (0.427, 0.30, -0.40, 0.0),   # declining
        (0.50,  0.29, -0.38, 0.0),
        (0.75,  0.27, -0.35, 0.0),
        (1.00,  0.26, -0.33, 0.0),
        (1.50,  0.25, -0.30, 0.0),
        (2.00,  0.24, -0.28, 0.0),
    ]

    slices = []
    for T, atm_vol, rho, _ in expiry_configs:
        F = spot * np.exp((r - q) * T)
        theta = atm_vol ** 2 * T
        psi = 0.5 + 0.1 * (T ** 0.5)  # mild psi term structure

        quotes = []
        for K in strikes_base:
            k = log(K / F)
            w = ssvi_w(k, theta, rho, psi)
            sigma = sqrt(w / T) if T > 0 else atm_vol
            from arbfree_vol.pricing.black_scholes import price_floats
            for otype in (OptionType.CALL, OptionType.PUT):
                price = price_floats(spot, K, T, r, q, sigma,
                                     is_call=(otype == OptionType.CALL))
                if price > 0.01:
                    quotes.append(
                        ExpirySlice.Quote if hasattr(ExpirySlice, 'Quote') else None
                    )
        # Build quotes properly
        from arbfree_vol.models.surface import Quote
        quotes = []
        for K in strikes_base:
            k = log(K / F)
            w = ssvi_w(k, theta, rho, psi)
            sigma = sqrt(w / T) if T > 0 else atm_vol
            from arbfree_vol.pricing.black_scholes import price_floats
            for otype in (OptionType.CALL, OptionType.PUT):
                price = price_floats(spot, K, T, r, q, sigma,
                                     is_call=(otype == OptionType.CALL))
                if price > 0.01:
                    quotes.append(Quote(strike=K, option_type=otype, price=price))

        sl = ExpirySlice(expiry_time=T, quotes=quotes)
        slices.append(sl)

    surface = VolSurface(spot=spot, risk_free=r, div_yield=q, slices=slices)
    return surface, []


# ── Slice data extraction ────────────────────────────────────────────


def extract_slice_data(surface: VolSurface) -> list[tuple[float, list[tuple[float, float]]]]:
    """Extract (T, [(k, w)]) from a VolSurface, same as the repair engine."""
    fwd_curve = estimate_forward_curve(surface)
    populate_per_slice_r(surface, fwd_curve)

    slices_data = []
    for sl in sorted(surface.slices, key=lambda s: s.expiry_time):
        F = fwd_curve.get(sl.expiry_time)
        if F is None:
            continue
        strike_w = slice_total_variance(surface, sl)
        if len(strike_w) < 5:
            continue
        pts = [(log(strike / F), w) for strike, w in strike_w.items()]
        pts.sort()
        slices_data.append((sl.expiry_time, pts))
    return slices_data


# ── Slice analytics ──────────────────────────────────────────────────


def compute_slice_analytics(T: float, points: list[tuple[float, float]]) -> dict:
    """Compute descriptive stats for a slice: ATM vol, skew measures."""
    ks = np.array([k for k, _ in points])
    ws = np.array([w for _, w in points])

    # ATM: interpolate w at k=0
    atm_w = float(np.interp(0.0, ks, ws))
    atm_vol = sqrt(atm_w / T) if T > 0 else 0.0

    # 25-delta skew (approximate: k ~ -0.6745 for 25-delta put, k ~ +0.6745 for call)
    k_25d_put = -0.6745
    k_25d_call = 0.6745
    w_25d_put = float(np.interp(k_25d_put, ks, ws))
    w_25d_call = float(np.interp(k_25d_call, ks, ws))
    skew_25d = (sqrt(w_25d_put / T) - sqrt(w_25d_call / T)) if T > 0 else 0.0

    # 10-delta skew (approximate: k ~ -1.2816 for 10-delta put)
    k_10d_put = -1.2816
    k_10d_call = 1.2816
    w_10d_put = float(np.interp(k_10d_put, ks, ws))
    w_10d_call = float(np.interp(k_10d_call, ks, ws))
    skew_10d = (sqrt(w_10d_put / T) - sqrt(w_10d_call / T)) if T > 0 else 0.0

    return {
        "T": T,
        "n_points": len(points),
        "atm_vol": atm_vol,
        "atm_w": atm_w,
        "skew_25d": skew_25d,
        "skew_10d": skew_10d,
        "k_range": (float(ks.min()), float(ks.max())),
    }


# ── Hard-constrained fit attempts ────────────────────────────────────


def try_hard_constrained(
    points: list[tuple[float, float]],
    prev: SSVIParams | None,
    label: str = "default",
) -> dict:
    """Try the hard-constrained _fit_slice; report success/failure and params."""
    try:
        params = _fit_slice(points, prev=prev)
        # Check constraint violations at the solution
        chi = params.theta * params.psi
        bf_resid = _butterfly_constraints(params.theta, params.rho, params.psi)

        violation_info = {
            "bf_min_residual": float(bf_resid.min()),
            "theta": params.theta,
            "rho": params.rho,
            "psi": params.psi,
            "chi": chi,
        }

        # Calendar constraint violations (if prev exists)
        if prev is not None:
            prev_chi = prev.theta * prev.psi
            violation_info["theta_delta"] = params.theta - prev.theta
            violation_info["chi_delta"] = chi - prev_chi
            denom = max(chi - prev_chi, 1e-6)
            ratio = (params.rho * chi - prev.rho * prev_chi) / denom
            violation_info["ratio"] = ratio
            violation_info["ratio_ok"] = abs(ratio) <= 1.0 + 1e-8

        return {
            "label": label,
            "converged": True,
            "params": params,
            "violations": violation_info,
        }
    except RuntimeError as e:
        return {
            "label": label,
            "converged": False,
            "params": None,
            "error": str(e),
        }


def try_warm_start(
    points: list[tuple[float, float]],
    prev: SSVIParams | None,
    warm_params: SSVIParams,
) -> dict:
    """Try hard-constrained fit warm-started from unconstrained solution.

    This is the key diagnostic: does the unconstrained solution provide
    a good initial point for the constrained optimizer?
    """
    # We need to temporarily monkeypatch the seed in _fit_slice.
    # Instead, let's directly call the optimizer with the warm-started x0.
    from scipy.optimize import minimize, NonlinearConstraint, Bounds

    ks = np.array([k for k, _ in points], dtype=np.float64)
    ws = np.array([w for _, w in points], dtype=np.float64)

    # Convert warm_params to internal (theta, u, v) representation
    theta0 = warm_params.theta
    rho0 = np.clip(warm_params.rho, -0.99, 0.99)
    p0 = np.clip(warm_params.psi, 1e-6, 20.0)
    u0 = float(np.arctanh(rho0))
    v0 = float(np.log(p0))
    x0 = np.array([theta0, u0, v0], dtype=np.float64)

    eps_theta = 1e-9
    eps_chi = 1e-6

    bounds = Bounds(
        lb=[1e-6, -6.0, float(np.log(1e-8))],
        ub=[10.0, 6.0, float(np.log(20.0))],
    )

    def _objective(x):
        theta, u, v = x
        rho = float(np.tanh(u))
        p = float(np.exp(v))
        return float(np.sum(
            (np.array([ssvi_w(float(k), theta, rho, p) for k in ks]) - ws) ** 2
        ))

    constraints = []

    def _bf_con(x):
        theta, u, v = x
        rho = float(np.tanh(u))
        p = float(np.exp(v))
        return _butterfly_constraints(theta, rho, p)

    constraints.append(NonlinearConstraint(_bf_con, 0.0, np.inf))

    if prev is not None:
        prev_chi = prev.theta * prev.psi

        def _theta_nd(x):
            return x[0] - prev.theta

        constraints.append(NonlinearConstraint(_theta_nd, eps_theta, np.inf))

        def _chi_nd(x):
            theta, u, v = x
            return theta * float(np.exp(v)) - prev_chi

        constraints.append(NonlinearConstraint(_chi_nd, eps_chi, np.inf))

        rho_prev_chi_prev = prev.rho * prev_chi

        def _ratio_upper(x):
            theta, u, v = x
            rho = float(np.tanh(u))
            chi = theta * float(np.exp(v))
            denom = max(chi - prev_chi, eps_chi)
            return (rho * chi - rho_prev_chi_prev) / denom

        def _ratio_lower(x):
            theta, u, v = x
            rho = float(np.tanh(u))
            chi = theta * float(np.exp(v))
            denom = max(chi - prev_chi, eps_chi)
            return -(rho * chi - rho_prev_chi_prev) / denom

        constraints.append(NonlinearConstraint(_ratio_upper, -1.0, 1.0))
        constraints.append(NonlinearConstraint(_ratio_lower, -1.0, 1.0))

    def _run(method, x_init, tol, maxiter):
        opts = {"maxiter": maxiter}
        if method == "trust-constr":
            opts["gtol"] = tol
        else:
            opts["ftol"] = tol
        return minimize(
            _objective, x_init, method=method, bounds=bounds,
            constraints=constraints, options=opts,
        )

    result = _run("trust-constr", x0, tol=1e-10, maxiter=500)
    success = result.success or (
        getattr(result, "status", -1) in (1, 2, 3)
    )

    if not success:
        result = _run("SLSQP", result.x, tol=1e-12, maxiter=1000)
        success = result.success or (
            getattr(result, "status", -1) in (0, 1, 2, 3)
        )

    if not success:
        return {
            "label": "warm-start",
            "converged": False,
            "params": None,
            "error": str(result.message),
            "final_objective": float(_objective(result.x)),
        }

    theta, u, v = result.x
    params = SSVIParams(
        theta=float(theta),
        rho=float(np.tanh(u)),
        psi=float(np.exp(v)),
    )

    chi = params.theta * params.psi
    bf_resid = _butterfly_constraints(params.theta, params.rho, params.psi)

    violation_info = {
        "bf_min_residual": float(bf_resid.min()),
        "theta": params.theta,
        "rho": params.rho,
        "psi": params.psi,
        "chi": chi,
    }

    if prev is not None:
        prev_chi = prev.theta * prev.psi
        violation_info["theta_delta"] = params.theta - prev.theta
        violation_info["chi_delta"] = chi - prev_chi
        denom = max(chi - prev_chi, 1e-6)
        ratio = (params.rho * chi - prev.rho * prev_chi) / denom
        violation_info["ratio"] = ratio
        violation_info["ratio_ok"] = abs(ratio) <= 1.0 + 1e-8

    return {
        "label": "warm-start",
        "converged": True,
        "params": params,
        "violations": violation_info,
        "final_objective": float(_objective(result.x)),
    }


def try_random_restarts(
    points: list[tuple[float, float]],
    prev: SSVIParams | None,
    n_restarts: int = 5,
) -> list[dict]:
    """Try hard-constrained fit with random initial seeds."""
    ws = np.array([w for _, w in points])
    w_min = float(ws.min())
    w_max = float(ws.max())

    results = []
    for i in range(n_restarts):
        rng = np.random.RandomState(42 + i * 7)

        # Random seed: theta in [w_min*0.5, w_min*2], rho in [-0.8, 0.1], psi in [0.1, 2.0]
        theta_seed = rng.uniform(w_min * 0.5, w_min * 2.0)
        rho_seed = rng.uniform(-0.8, 0.1)
        psi_seed = rng.uniform(0.1, 2.0)

        # Adjust for calendar constraints if prev exists
        if prev is not None:
            prev_chi = prev.theta * prev.psi
            theta_seed = max(prev.theta + 1e-9, theta_seed)
            chi_seed = theta_seed * psi_seed
            if chi_seed < prev_chi + 1e-6:
                psi_seed = (prev_chi + 1e-6) / theta_seed

        rho_seed = float(np.clip(rho_seed, -0.99, 0.99))
        psi_seed = float(np.clip(psi_seed, 1e-6, 20.0))

        # Try unconstrained least_squares for this seed first, then use as warm-start
        from scipy.optimize import least_squares as _ls

        def _seed_resid(p):
            th, rh, ps = p
            return np.array([
                ssvi_w(float(k), th, rh, ps) - float(w)
                for k, w in points
            ])

        seed_result = _ls(
            _seed_resid,
            x0=[theta_seed, rho_seed, psi_seed],
            bounds=([1e-6, -0.999, 1e-6], [10.0, 0.999, 20.0]),
        )
        seed_params = SSVIParams(
            theta=float(seed_result.x[0]),
            rho=float(seed_result.x[1]),
            psi=float(seed_result.x[2]),
        )

        r = try_hard_constrained(points, prev, label=f"restart-{i}")
        if not r["converged"]:
            # Try with this random seed's unconstrained result as warm-start
            r = try_warm_start(points, prev, seed_params)
            r["label"] = f"restart-{i}-warm"
        results.append(r)

    return results


# ── Check if unconstrained params satisfy H&M with neighbors ────────


def check_unconstrained_satisfies_hm(
    all_params: list[tuple[float, SSVIParams]],
    fallback_T: float,
    unconstrained_params: SSVIParams,
) -> dict:
    """Check if the unconstrained fit's params satisfy H&M conditions
    with both the predecessor and successor slices.

    Returns a dict with:
    - satisfies_with_prev: bool
    - satisfies_with_next: bool
    - satisfies_both: bool
    - details: dict with the constraint values
    """
    # Find predecessor and successor
    sorted_params = sorted(all_params, key=lambda x: x[0])
    T_sorted = [T for T, _ in sorted_params]

    idx = None
    for i, (T, _) in enumerate(sorted_params):
        if abs(T - fallback_T) < 1e-6:
            idx = i
            break

    result = {
        "satisfies_with_prev": True,
        "satisfies_with_next": True,
        "satisfies_both": True,
        "details": {},
    }

    if idx is None:
        result["satisfies_both"] = False
        return result

    # Check with predecessor
    if idx > 0:
        prev_T, prev_p = sorted_params[idx - 1]
        # Build params list with [prev, unconstrained] and check
        test_params = [prev_p, unconstrained_params]
        hm_ok = verify_hm_condition(test_params)
        result["satisfies_with_prev"] = hm_ok

        # Compute the actual constraint values
        prev_chi = prev_p.theta * prev_p.psi
        my_chi = unconstrained_params.theta * unconstrained_params.psi
        denom = max(my_chi - prev_chi, 1e-6)
        ratio = (unconstrained_params.rho * my_chi - prev_p.rho * prev_chi) / denom
        result["details"]["prev_pair"] = {
            "prev_T": prev_T,
            "theta_delta": unconstrained_params.theta - prev_p.theta,
            "chi_delta": my_chi - prev_chi,
            "ratio": ratio,
            "ratio_ok": abs(ratio) <= 1.0 + 1e-8,
        }

    # Check with successor
    if idx < len(sorted_params) - 1:
        next_T, next_p = sorted_params[idx + 1]
        test_params = [unconstrained_params, next_p]
        hm_ok = verify_hm_condition(test_params)
        result["satisfies_with_next"] = hm_ok

        my_chi = unconstrained_params.theta * unconstrained_params.psi
        next_chi = next_p.theta * next_p.psi
        denom = max(next_chi - my_chi, 1e-6)
        ratio = (next_p.rho * next_chi - unconstrained_params.rho * my_chi) / denom
        result["details"]["next_pair"] = {
            "next_T": next_T,
            "theta_delta": next_p.theta - unconstrained_params.theta,
            "chi_delta": next_chi - my_chi,
            "ratio": ratio,
            "ratio_ok": abs(ratio) <= 1.0 + 1e-8,
        }

    result["satisfies_both"] = (
        result["satisfies_with_prev"] and result["satisfies_with_next"]
    )
    return result


# ── Main diagnostic ──────────────────────────────────────────────────


def run_diagnostics():
    """Run the full diagnostic pipeline."""
    print("=" * 72)
    print("  Fallback Slice Diagnostic: SPY eSSVI term structure")
    print("=" * 72)

    # Step 1: Fetch data
    print("\n[1/4] Fetching SPY data...")
    surface, rejected = fetch_spy_data()
    print(f"  Spot: {surface.spot:.2f}")
    print(f"  Slices: {len(surface.slices)}")
    print(f"  Quotes: {sum(len(s.quotes) for s in surface.slices)}")
    if rejected:
        print(f"  Rejected: {len(rejected)}")

    # Step 2: Extract slice data
    print("\n[2/4] Extracting (k, w) data...")
    slices_data = extract_slice_data(surface)
    print(f"  Extracted {len(slices_data)} slices")
    for T, pts in slices_data:
        print(f"    T={T:.4f}: {len(pts)} points")

    # Step 3: Run sequential fit
    print("\n[3/4] Running sequential eSSVI fit...")
    result = fit_ssvi_surface_sequential(slices_data)

    print(f"  Fitted: {len(result.fitted_slices)} slices")
    print(f"  Fallback: {len(result.fallback_slices)} slices")
    if result.fallback_slices:
        print(f"    T = {[f'{T:.4f}' for T in result.fallback_slices]}")
    print(f"  Failed: {len(result.failed_slices)} slices")
    if result.failed_slices:
        print(f"    T = {[f'{T:.4f}' for T in result.failed_slices]}")

    if not result.fallback_slices:
        print("\n  No fallback slices found. Nothing to diagnose.")
        return

    # Step 4: Detailed diagnostics for each fallback slice
    print("\n[4/4] Detailed diagnostics for fallback slices...")
    print("=" * 72)

    # Build a map of all fitted params by T
    fitted_by_T = {T: p for T, p in result.fitted_slices}

    # Build a map of all slices_data by T
    data_by_T = {T: pts for T, pts in slices_data}

    summary_rows = []

    for fallback_T in result.fallback_slices:
        print(f"\n{'-' * 72}")
        print(f"  FALLBACK SLICE: T = {fallback_T:.4f}")
        print(f"{'-' * 72}")

        pts = data_by_T.get(fallback_T)
        if pts is None:
            print("  ERROR: No data found for this T")
            continue

        # (a) Slice analytics
        analytics = compute_slice_analytics(fallback_T, pts)
        print(f"\n  Slice data:")
        print(f"    T          = {analytics['T']:.4f}")
        print(f"    n_points   = {analytics['n_points']}")
        print(f"    ATM vol    = {analytics['atm_vol']:.4f}")
        print(f"    ATM w      = {analytics['atm_w']:.6f}")
        print(f"    25d skew   = {analytics['skew_25d']:.4f}")
        print(f"    10d skew   = {analytics['skew_10d']:.4f}")
        print(f"    k range    = [{analytics['k_range'][0]:.3f}, {analytics['k_range'][1]:.3f}]")

        # (b) Unconstrained fit
        print(f"\n  Unconstrained fit (fit_ssvi_slice):")
        try:
            unc_params = fit_ssvi_slice(pts)
            unc_rmse = sqrt(np.mean([
                (ssvi_w(k, unc_params.theta, unc_params.rho, unc_params.psi) - w) ** 2
                for k, w in pts
            ]))
            print(f"    theta = {unc_params.theta:.6f}")
            print(f"    rho   = {unc_params.rho:.6f}")
            print(f"    psi   = {unc_params.psi:.6f}")
            print(f"    chi   = {unc_params.theta * unc_params.psi:.6f}")
            print(f"    RMSE  = {unc_rmse:.8f}")
        except RuntimeError as e:
            print(f"    FAILED: {e}")
            unc_params = None
            unc_rmse = float("nan")

        # (c) Find predecessor (last hard-constrained fit before this T)
        prev_params = None
        sorted_fitted = sorted(result.fitted_slices, key=lambda x: x[0])
        for T_i, p_i in sorted_fitted:
            if T_i < fallback_T:
                prev_params = p_i
            else:
                break

        if prev_params:
            print(f"\n  Predecessor slice (last hard-constrained fit):")
            prev_T = [T for T, p in sorted_fitted if p is prev_params][0]
            print(f"    T     = {prev_T:.4f}")
            print(f"    theta = {prev_params.theta:.6f}")
            print(f"    rho   = {prev_params.rho:.6f}")
            print(f"    psi   = {prev_params.psi:.6f}")
            print(f"    chi   = {prev_params.theta * prev_params.psi:.6f}")
        else:
            print(f"\n  No predecessor (this is the first slice or all predecessors are fallback)")

        # (d) Hard-constrained fit attempts
        print(f"\n  Hard-constrained fit attempts:")

        # Default seed
        default_result = try_hard_constrained(pts, prev_params, label="default")
        print(f"\n    [default seed]")
        if default_result["converged"]:
            v = default_result["violations"]
            print(f"      CONVERGED")
            print(f"      theta = {v['theta']:.6f}, rho = {v['rho']:.6f}, psi = {v['psi']:.6f}")
            print(f"      chi = {v['chi']:.6f}, bf_min_resid = {v['bf_min_residual']:.6f}")
            if prev_params and "ratio" in v:
                print(f"      theta_delta = {v['theta_delta']:.6f}, chi_delta = {v['chi_delta']:.6f}")
                print(f"      ratio = {v['ratio']:.6f}, ratio_ok = {v['ratio_ok']}")
        else:
            print(f"      FAILED: {default_result.get('error', 'unknown')}")

        # Warm-start from unconstrained
        if unc_params is not None:
            warm_result = try_warm_start(pts, prev_params, unc_params)
            print(f"\n    [warm-start from unconstrained]")
            if warm_result["converged"]:
                v = warm_result["violations"]
                print(f"      CONVERGED")
                print(f"      theta = {v['theta']:.6f}, rho = {v['rho']:.6f}, psi = {v['psi']:.6f}")
                print(f"      chi = {v['chi']:.6f}, bf_min_resid = {v['bf_min_residual']:.6f}")
                if prev_params and "ratio" in v:
                    print(f"      theta_delta = {v['theta_delta']:.6f}, chi_delta = {v['chi_delta']:.6f}")
                    print(f"      ratio = {v['ratio']:.6f}, ratio_ok = {v['ratio_ok']}")
                print(f"      final_objective = {warm_result['final_objective']:.8f}")
            else:
                print(f"      FAILED: {warm_result.get('error', 'unknown')}")
                print(f"      final_objective = {warm_result.get('final_objective', 'N/A')}")
        else:
            warm_result = {"converged": False}

        # Random restarts
        restart_results = try_random_restarts(pts, prev_params, n_restarts=5)
        n_restart_converged = sum(1 for r in restart_results if r["converged"])
        print(f"\n    [5 random restarts]")
        print(f"      Converged: {n_restart_converged} / 5")
        for r in restart_results:
            if r["converged"]:
                v = r["violations"]
                print(f"        {r['label']}: theta={v['theta']:.6f}, rho={v['rho']:.6f}, "
                      f"psi={v['psi']:.6f}, bf_min={v['bf_min_residual']:.6f}")
            else:
                print(f"        {r['label']}: FAILED")

        # (e) Check if unconstrained params satisfy H&M with neighbors
        print(f"\n  Does the unconstrained fit satisfy H&M with neighbors?")
        if unc_params is not None:
            hm_check = check_unconstrained_satisfies_hm(
                result.fitted_slices, fallback_T, unc_params
            )
            print(f"    satisfies_with_prev  = {hm_check['satisfies_with_prev']}")
            print(f"    satisfies_with_next  = {hm_check['satisfies_with_next']}")
            print(f"    satisfies_both       = {hm_check['satisfies_both']}")

            for pair_name, pair_info in hm_check["details"].items():
                print(f"    {pair_name}:")
                for k, v in pair_info.items():
                    print(f"      {k} = {v}")
        else:
            hm_check = {"satisfies_both": False}

        # Summary row
        summary_rows.append({
            "T": fallback_T,
            "unc_rmse": unc_rmse,
            "default_converged": default_result["converged"],
            "warm_start_converged": warm_result["converged"],
            "restart_converged": n_restart_converged,
            "unc_satisfies_hm": hm_check["satisfies_both"],
        })

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 72}")
    print(f"\n  {'T':>8}  {'unc-RMSE':>10}  {'default':>8}  {'warm-st':>8}  "
          f"{'restart':>8}  {'unc-HM':>8}")
    print(f"  {'-' * 58}")
    for row in summary_rows:
        print(f"  {row['T']:>8.4f}  {row['unc_rmse']:>10.6f}  "
              f"{'YES' if row['default_converged'] else 'NO':>8}  "
              f"{'YES' if row['warm_start_converged'] else 'NO':>8}  "
              f"{row['restart_converged']:>5}/5   "
              f"{'YES' if row['unc_satisfies_hm'] else 'NO':>8}")

    # ── Interpretation ───────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  INTERPRETATION")
    print(f"{'=' * 72}")

    convergence_failures = [r for r in summary_rows
                           if not r["default_converged"]]
    warm_fixes = [r for r in summary_rows
                  if not r["default_converged"] and r["warm_start_converged"]]
    fundamental = [r for r in summary_rows
                   if not r["default_converged"] and not r["unc_satisfies_hm"]]

    print(f"\n  Total fallback slices: {len(summary_rows)}")
    print(f"  Convergence failures (default seed): {len(convergence_failures)}")
    print(f"  Warm-start fixes: {len(warm_fixes)}")
    print(f"  Fundamental infeasibility: {len(fundamental)}")

    if len(warm_fixes) >= 2:
        print(f"\n  => OUTCOME A: Warm-start from unconstrained fit fixes {len(warm_fixes)} slices.")
        print(f"     The hard-constrained fit has a convergence problem, not a fundamental issue.")
        print(f"     Recommended: implement 2-stage strategy (default seed -> warm-start).")
    elif len(fundamental) >= 2:
        print(f"\n  => OUTCOME B: Unconstrained fit's params don't satisfy H&M with neighbors.")
        print(f"     SPY data has a term-structure feature the SSVI parametrization can't represent.")
        print(f"     Recommended: document as known limitation in docs/issues.md.")
    else:
        print(f"\n  => OUTCOME C: Mixed results.")
        print(f"     Some slices are convergence failures (warm-start would fix them).")
        print(f"     Others are fundamental infeasibility (document as limitation).")

    return summary_rows


# ── Entry point ──────────────────────────────────────────────────────


if __name__ == "__main__":
    summary = run_diagnostics()
