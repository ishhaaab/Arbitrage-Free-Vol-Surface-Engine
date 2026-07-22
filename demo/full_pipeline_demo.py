"""Arbfree-vol-surface full pipeline demo.

Showcases every major module on synthetic data (no network required):

  1.  Build a 2-slice clean VolSurface from known SVI smiles.
  2.  Repair with SVI, eSSVI, and SABR — compare fit quality.
  3.  Build a FittedSurface, query iv_at on a (K,T) grid.
  4.  Portfolio Greeks and spot-bump scenario.
  5.  Dupire local volatility on the fitted surface.
  6.  Surface-dynamics PCA on a synthetic time series.

Run from the repo root:  python demo/full_pipeline_demo.py
"""

from datetime import date, timedelta
from math import sqrt
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.svi.model import SVIParams, svi_total_variance
from arbfree_vol.repair.engine import repair
from arbfree_vol.surface.interpolate import build_fitted_surface, iv_at
from arbfree_vol.surface.greeks import portfolio_greeks
from arbfree_vol.surface.risk import spot_bump_analysis, parallel_vega_pnl
from arbfree_vol.pricing.local_vol import dupire
from arbfree_vol.dynamics import (
    SurfaceSnapshot, SurfaceSeries, parameter_matrix, pca_deformations,
    principal_mode_labels,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SPOT = 100.0
R = 0.05
Q = 0.01
_DUMMY_DATE = date(2030, 1, 1)


def _bs_price(otype: OptionType, strike: float, sigma: float,
              tt: float, spot: float = SPOT) -> float:
    """Black-Scholes price (float-level)."""
    from arbfree_vol.pricing.black_scholes import price_floats
    return price_floats(spot, strike, tt, R, Q, sigma,
                        is_call=(otype == OptionType.CALL))


def _quote(K: float, otype: OptionType, sigma: float, tt: float) -> Quote:
    """One option quote priced at a given Black vol."""
    return Quote(strike=K, option_type=otype,
                 price=_bs_price(otype, K, sigma, tt))


def _svi_quotes(params: SVIParams, expiry: float,
                strikes: list[float]) -> list[Quote]:
    """Build call + put quotes for each strike priced at the SVI vol."""
    quotes = []
    F = SPOT * np.exp((R - Q) * expiry)
    for K in strikes:
        k = np.log(K / F)
        w = svi_total_variance(k, params.a, params.b, params.rho,
                               params.m, params.sigma)
        sigma = sqrt(w / expiry)
        for otype in (OptionType.CALL, OptionType.PUT):
            quotes.append(_quote(K, otype, sigma, expiry))
    return quotes


# ---------------------------------------------------------------------------
# Section helpers
# ---------------------------------------------------------------------------

def _sep(title: str) -> None:
    n = 72
    print()
    print("=" * n)
    print(f"  {title}")
    print("=" * n)


def _sub(title: str) -> None:
    print(f"\n--- {title} ---")


# ===================================================================
# 1.  Synthetic surface
# ===================================================================

@dataclass
class DemoSurface:
    surface: VolSurface
    true_params: dict[float, SVIParams]
    expiry_times: list[float]
    strikes: list[float]


def build_demo_surface() -> DemoSurface:
    """A 2-slice surface with parameterised SVI smiles."""
    strikes = [80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0]
    T1, T2 = 0.5, 2.0
    F1 = SPOT * np.exp((R - Q) * T1)
    F2 = SPOT * np.exp((R - Q) * T2)

    # --- Slice 1 (short-dated): mild skew ---
    p1 = SVIParams(a=0.02, b=0.3, rho=-0.3, m=0.0, sigma=0.12)
    q1 = _svi_quotes(p1, T1, strikes)

    # --- Slice 2 (long-dated): more pronounced skew ---
    p2 = SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.05, sigma=0.15)
    q2 = _svi_quotes(p2, T2, strikes)

    surface = VolSurface(
        spot=SPOT, risk_free=R, div_yield=Q,
        slices=[
            ExpirySlice(expiry_time=T1, quotes=q1),
            ExpirySlice(expiry_time=T2, quotes=q2),
        ],
    )
    return DemoSurface(
        surface=surface,
        true_params={T1: p1, T2: p2},
        expiry_times=[T1, T2],
        strikes=strikes,
    )


# ===================================================================
# 2.  Repair comparison
# ===================================================================

def run_repair_comparison(demo: DemoSurface) -> None:
    _sep("Repair: SVI vs eSSVI vs SABR")

    results = {}
    for label, kw in [("SVI",   {"use_ssvi": False, "use_sabr": False}),
                      ("eSSVI",  {"use_ssvi": True,  "use_sabr": False}),
                      ("SABR",  {"use_ssvi": False,  "use_sabr": True})]:
        report = repair(demo.surface, **kw)
        results[label] = report

    # Summary table
    print(f"\n  {'Model':<8} {'Slices':>7} {'Rejected':>9} {'Fit RMSE':>9} "
          f"{'Violations':>11} {'Viol. after':>11}")
    print(f"  {'-'*56}")
    for label, report in results.items():
        rmses = [s.rmse for s in report.fitted_slices]
        avg_rmse = np.mean(rmses) if rmses else float("nan")
        n_slices = report.metrics.n_slices_fitted
        print(f"  {label:<8} {n_slices:>7} {report.metrics.n_rejected:>9} "
              f"{avg_rmse:>9.3e} "
              f"{report.metrics.n_violations_before:>9} "
              f"{report.metrics.n_violations_after:>9}")
    print()

    # Compare SVI vs SABR fitted params for slice 1
    rep_svi = results["SVI"]
    rep_sabr = results["SABR"]
    if rep_svi.fitted_slices and rep_sabr.fitted_slices:
        _sub("Fitted parameters for T=0.5 slice")
        p_true = demo.true_params[0.5]
        p_svi = rep_svi.fitted_slices[0].params
        p_sabr = rep_sabr.fitted_slices[0].params
        print(f"  {'Param':<10} {'True':>10} {'SVI fit':>10} {'SABR->SVI':>10}")
        print(f"  {'-'*40}")
        for name in ("a", "b", "rho", "m", "sigma"):
            t = getattr(p_true, name)
            svi = getattr(p_svi, name)
            sabr = getattr(p_sabr, name)
            print(f"  {name:<10} {t:>10.4f} {svi:>10.4f} {sabr:>10.4f}")


# ===================================================================
# 3.  iv_at query grid
# ===================================================================

def run_iv_at_grid(demo: DemoSurface) -> None:
    _sep("FittedSurface query: iv_at on a (K,T) grid")

    # Use the SVI repair report to build a FittedSurface
    report = repair(demo.surface)
    fs = build_fitted_surface(report)

    query_strikes = [85.0, 95.0, 100.0, 105.0, 115.0]
    query_maturities = [0.5, 0.75, 1.0, 1.5, 2.0]

    print(f"\n  'K\\T'  ", end="")
    for T in query_maturities:
        print(f" {T:>7.2f}", end="")
    print()
    for K in query_strikes:
        print(f"  {K:>5.1f}  ", end="")
        for T in query_maturities:
            try:
                iv = iv_at(fs, K, T)
            except ValueError:
                iv = float("nan")
            print(f" {iv:>7.4f}", end="")
        print()
    print()

    # Also show the forward curve used
    _sub("Forward curve (per slice)")
    for expiry, fwd in fs.forward_curve:
        print(f"  T={expiry:.2f}  F={fwd:.4f}")


# ===================================================================
# 4.  Portfolio Greeks + scenario analysis
# ===================================================================

def run_portfolio_risk(demo: DemoSurface) -> None:
    _sep("Portfolio Greeks and spot-bump scenario")

    report = repair(demo.surface)
    fs = build_fitted_surface(report)

    from arbfree_vol.models.option import OptionContract

    # Hypothetical portfolio: 1 long ATM call + 1 short OTM put
    positions = [
        (OptionContract(symbol="SPY", option_type=OptionType.CALL,
                        strike=100.0, expiry_date=_DUMMY_DATE),
         0.5, 1.0),   # long 1 call at T=0.5
        (OptionContract(symbol="SPY", option_type=OptionType.PUT,
                        strike=95.0, expiry_date=_DUMMY_DATE),
         1.0, -0.5),  # short 0.5 puts at T=1.0
    ]

    # Greeks
    greeks = portfolio_greeks(fs, positions, r=R, q=Q)
    print(f"\n  Portfolio Greeks (aggregate):")
    print(f"    Delta: {greeks.total_delta:>10.4f}")
    print(f"    Gamma: {greeks.total_gamma:>10.4f}")
    print(f"    Vega:  {greeks.total_vega:>10.4f}")
    print(f"    Theta: {greeks.total_theta:>10.4f}")
    print(f"    Rho:   {greeks.total_rho:>10.4f}")

    # Spot-bump scenarios
    bumps = [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10]
    scenarios = spot_bump_analysis(fs, positions, bumps)
    print(f"\n  Spot bump scenarios:")
    print(f"    {'Bump':>8} {'PnL':>10}")
    for sc in scenarios:
        print(f"    {sc.spot_bump:>+7.0%} {sc.pnl:>+10.4f}")

    # Vega parallel shift
    pnl = parallel_vega_pnl(fs, positions, vega_shift=0.01)
    print(f"\n  Parallel vega P&L for +1 vol point: {pnl:>+.4f}")


# ===================================================================
# 5.  Dupire local volatility
# ===================================================================

def run_dupire(demo: DemoSurface) -> None:
    _sep("Dupire local volatility (flat-vol benchmark)")

    # Build a FittedSurface with FLAT smiles (b=0) for the benchmark.
    sigma_flat = 0.25
    T1, T2 = 0.5, 2.0

    flat_sl1 = repair(demo.surface).fitted_slices[0]
    flat_sl2 = repair(demo.surface).fitted_slices[1]

    # Build flat-smile slices: w(k) = a = sigma^2 * T, b=0.
    # SVI sigma_param is unused when b=0 (any positive value works).
    flat_sl1_fwd = SPOT * np.exp((R - Q) * T1)
    flat_sl2_fwd = SPOT * np.exp((R - Q) * T2)
    flat_sl1_flat = type(flat_sl1)(
        expiry_time=T1,
        params=SVIParams(a=sigma_flat ** 2 * T1, b=0.0,
                          rho=0.0, m=0.0, sigma=0.1),
        rmse=0.0, forward_price=flat_sl1_fwd,
        n_quotes_total=10, n_quotes_used=10,
    )
    flat_sl2_flat = type(flat_sl2)(
        expiry_time=T2,
        params=SVIParams(a=sigma_flat ** 2 * T2, b=0.0,
                          rho=0.0, m=0.0, sigma=0.1),
        rmse=0.0, forward_price=flat_sl2_fwd,
        n_quotes_total=10, n_quotes_used=10,
    )

    from arbfree_vol.surface.interpolate import FittedSurface
    fs_flat = FittedSurface(
        spot=SPOT, risk_free=R, div_yield=Q,
        forward_curve=((T1, flat_sl1_fwd), (T2, flat_sl2_fwd)),
        fitted_slices=(flat_sl1_flat, flat_sl2_flat),
    )

    strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
    maturities = [T1, 1.0, 1.5, T2]

    lv = dupire(fs_flat, strikes, maturities)

    print(f"\n  Flat-vol benchmark (target sigma={sigma_flat} everywhere)")
    print(f"  'K\\T'  ", end="")
    for T in maturities:
        print(f" {T:>7.2f}", end="")
    print()
    for iK, K in enumerate(strikes):
        print(f"  {K:>5.1f}  ", end="")
        for iT in range(len(maturities)):
            v = lv.grid[iT][iK]
            print(f" {v:>7.4f}", end="")
        print()
    print(f"\n  Interior rows (T=1.0, 1.5) should be ~{sigma_flat} "
          f"— flat-vol recovery validation.")


# ===================================================================
# 6.  Surface dynamics PCA
# ===================================================================

def run_pca_demo(demo: DemoSurface) -> None:
    _sep("Surface dynamics: PCA on synthetic time series")

    # Synthetic 30-snapshot series: rho drifts, sigma oscillates
    n = 30
    base_params = demo.true_params[0.5]
    snapshot_template = repair(demo.surface).fitted_slices[0]
    snapshots = []
    for i in range(n):
        t = i / (n - 1)  # 0..1
        rho = -0.5 + 0.4 * t
        sigma_p = 0.10 + 0.08 * np.sin(2 * np.pi * t)

        sl = type(snapshot_template)(
            expiry_time=0.5,
            params=SVIParams(a=base_params.a, b=base_params.b,
                              rho=rho, m=base_params.m, sigma=sigma_p),
            rmse=0.02, forward_price=SPOT * np.exp((R - Q) * 0.5),
            n_quotes_total=14, n_quotes_used=14,
        )
        snapshots.append(
            SurfaceSnapshot(
                snapshot_date=date(2024, 1, 1) + timedelta(days=i),
                fitted_slices=(sl,),
            )
        )

    series = SurfaceSeries(tuple(snapshots))
    mat, buckets, labels = parameter_matrix(series)

    result = pca_deformations(mat, n_components=3)

    print(f"\n  Parameter matrix shape: {mat.shape}")
    print(f"  Expiry buckets: {buckets}")
    print(f"  PCA components requested: 3")
    print(f"  Explained variance ratios:")
    for i, ev in enumerate(result.explained_variance_ratio):
        label = principal_mode_labels(i + 1)[-1]
        cumulative = sum(result.explained_variance_ratio[:i + 1])
        print(f"    PC{i + 1} ({label:<10}): {ev:>7.1%}  (cumulative: {cumulative:>7.1%})")


# ===================================================================
# Main
# ===================================================================

def main():
    print()
    print(" " + "=" * 62)
    print("   arbfree-vol-surface  --  Full Pipeline Demo")
    print("   Synthetic data (no network required)")
    print(" " + "=" * 62)

    demo = build_demo_surface()

    print(f"\n  Spot={SPOT}, r={R}, q={Q}")
    print(f"  Strikes: {demo.strikes}")
    print(f"  Expiries: {demo.expiry_times}")

    run_repair_comparison(demo)
    run_iv_at_grid(demo)
    run_portfolio_risk(demo)
    run_dupire(demo)
    run_pca_demo(demo)

    _sep("Done")
    print(
        "  All modules exercised successfully.\n"
        "  The full test suite can be run with:  pytest tests/ -q\n"
    )


if __name__ == "__main__":
    main()
