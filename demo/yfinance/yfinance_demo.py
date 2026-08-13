"""Demo: yfinance -> repair with SVI/eSSVI/SABR -> fitted surface -> Greeks -> Dupire -> 7 plots.

Uses the ``ingestion.yahoo`` module which sources real risk-free
rates (^IRX) and dividend yields, fetches mid prices, and applies
the cleaning layer before building the surface.

Usage::

    python demo/yfinance/yfinance_demo.py                # default: SPY
    python demo/yfinance/yfinance_demo.py --symbol QQQ   # any US equity/ETF
"""

import matplotlib
matplotlib.use("Agg")  # save to files, no GUI

# yfinance is a soft dependency; gracefully exit if missing.
try:
    import yfinance as yf  # noqa: F401
except ImportError:
    print("yfinance is required.  Install with:  pip install yfinance")
    raise SystemExit(1)

import argparse
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

from arbfree_vol.ingestion.yahoo import fetch_chain
from arbfree_vol.repair.engine import repair
from arbfree_vol.surface.interpolate import build_fitted_surface
from arbfree_vol.pricing.local_vol import dupire
from arbfree_vol.models.option import OptionContract, OptionType

# ##########################################################################
# 0. Parse CLI args
# ##########################################################################
parser = argparse.ArgumentParser(
    description="yfinance -> repair with SVI/eSSVI/SABR -> fitted surface -> Greeks -> Dupire -> plots",
)
parser.add_argument("--symbol", default="SPY", help="Ticker symbol, e.g. SPY, QQQ, AAPL, MSFT")
args = parser.parse_args()
symbol = args.symbol
_OUT = Path(__file__).parent

# ##########################################################################
# 1. Fetch + clean
# ##########################################################################
print(f"Fetching {symbol} chain (mid prices, real r/q, with cleaning)...")
surface, rejected, quality_drops = fetch_chain(symbol, max_expiries=20, min_T_years=7.0 / 365.0)

T_count = len(surface.slices)
Q_count = sum(len(s.quotes) for s in surface.slices)

# Quote accounting, each quantity with its explicit denominator:
#   raw rows     = rows the data-quality filter saw (quality drops + rows
#                  that survived the filter with a price).
#   quotes built = rows that survived the filter AND produced a Quote
#                  (cleaning rejects + retained quotes).
# Rows whose bid/ask/last-price are ALL absent are dropped silently by the
# fetcher (_row_to_quote returns None) and appear in none of the returned
# lists; in practice yfinance supplies price fields for every row, so the
# raw-row denominator below is exact for this source.
n_raw = len(quality_drops) + len(rejected) + Q_count
n_quotes_built = len(rejected) + Q_count
print(f"  Raw rows fetched: {n_raw}")
print(f"    Dropped by data quality filter: {len(quality_drops)} of {n_raw} raw rows "
      f"({len(quality_drops) / max(n_raw, 1) * 100:.1f}%)")
print(f"    Quotes built from surviving rows: {n_quotes_built} of {n_raw} raw rows "
      f"({n_quotes_built / max(n_raw, 1) * 100:.1f}%)")
print(f"    Rejected by cleaning: {len(rejected)} of {n_quotes_built} quotes built "
      f"({len(rejected) / max(n_quotes_built, 1) * 100:.1f}%)")
print(f"    Retained quotes: {Q_count} of {n_quotes_built} quotes built "
      f"({Q_count / max(n_quotes_built, 1) * 100:.1f}%)")
if quality_drops:
    dq_reasons = Counter()
    for d in quality_drops:
        for part in d.reason.split("; "):
            dq_reasons[part.split("=")[0]] += 1
    print(f"  Quality drop breakdown:")
    for reason, count in dq_reasons.most_common():
        print(f"    {reason}: {count}")
print(f"  Expiries: {T_count}")
print(f"  Spot={surface.spot:.2f}, r={surface.risk_free:.4f}, "
      f"q={surface.div_yield:.4f}")
if rejected:
    rule_counts = Counter(r.rule.value for r in rejected)
    print(f"  Rejection breakdown (first rule hit per quote):")
    for rule, count in rule_counts.most_common():
        print(f"    {rule}: {count} ({count / len(rejected) * 100:.1f}%)")
    if "zero_price" in rule_counts:
        print(f"  (zero_price = yfinance 'no quote' rows with bid=ask=0; "
              f"not real violations)")

# ##########################################################################
# 2. Repair with all 3 models
# ##########################################################################
print("Repairing with SVI, eSSVI, SABR...")
reports: dict[str, object] = {}
model_configs = [
    ("SVI", {"use_ssvi": False, "use_sabr": False}),
    ("eSSVI", {"use_ssvi": True, "use_sabr": False}),
    ("SABR", {"use_ssvi": False, "use_sabr": True}),
]
for label, kw in model_configs:
    r = repair(surface, **kw)
    reports[label] = r
    n_before = r.metrics.n_violations_before
    n_after = r.metrics.n_violations_after
    n_rej = r.metrics.n_rejected
    avg_rmse = (
        sum(fs.rmse for fs in r.fitted_slices) / len(r.fitted_slices)
        if r.fitted_slices else 0.0
    )
    print(f"  {label:6s}: violations {n_before} -> {n_after}, "
          f"rejected={n_rej}, fitted={len(r.fitted_slices)}, "
          f"avg_RMSE={avg_rmse:.4f}")

    # Print eSSVI-specific fallback/failed slice info
    if label == "eSSVI":
        fb = r.fallback_slices
        fl = r.failed_slices
        if fb:
            print(f"  Fallback slices (hard-constrained failed, "
                  f"unconstrained succeeded): {len(fb)}")
            print(f"    T = {fb}")
            print(f"  Note: these slices do not satisfy the "
                  f"H&M Prop 3.1 arb-free condition.")
        else:
            print(f"  No fallback slices — all expiries "
                  f"arb-free-by-construction.")
        print(f"  Failed slices (both fits failed): {len(fl)}")
        if fl:
            print(f"    T = {fl}")

# ##########################################################################
# 3. Build FittedSurface from the eSSVI report + extract its fallback slices
# ##########################################################################
# The IV / Dupire / Greeks plots are annotated with eSSVI fallback
# diagnostics, so the plotted surface must come from the SAME report that
# supplies those diagnostics (not the raw-SVI one).  Otherwise the fallback
# rows would be grayed out on a surface that never fell back at those
# maturities — the masking would annotate a different model's fit.
essvi_report = reports["eSSVI"]
fs = build_fitted_surface(essvi_report)

# Extract fallback T values from the eSSVI report for plot masking.
# These are maturities where the hard-constrained H&M Prop 3.1 fit failed
# and the unconstrained per-slice fallback was used instead.
fallback_Ts: list[float] = essvi_report.fallback_slices
if fallback_Ts:
    print(f"  eSSVI fallback T values (will be grayed in heatmaps): "
          f"{[f'{T:.4f}' for T in fallback_Ts]}")
else:
    print(f"  No eSSVI fallback slices — all plots show arb-free data.")

# ##########################################################################
# 4. Dupire local vol
# ##########################################################################
spot = surface.spot
strikes = list(np.linspace(spot * 0.85, spot * 1.15, 20))

# Maturities: base from surface, then pad to >= 3 points.
mat_base = sorted(sl.expiry_time for sl in fs.fitted_slices)
if len(mat_base) < 3:
    # Interleave mid-points to get a dense enough grid for dupire()
    maturities = []
    for i in range(len(mat_base) - 1):
        T1 = mat_base[i]
        T2 = mat_base[i + 1]
        maturities.append(T1)
        maturities.append((T1 + T2) / 2.0)
    maturities.append(mat_base[-1])
    # Deduplicate while preserving order
    seen: set[float] = set()
    maturities_dedup: list[float] = []
    for t in maturities:
        rounded = round(t, 10)
        if rounded not in seen:
            seen.add(rounded)
            maturities_dedup.append(t)
    maturities = maturities_dedup
else:
    maturities = mat_base

print(f"  Building Dupire grid: {len(strikes)} strikes x {len(maturities)} maturities")
lv = dupire(fs, strikes, maturities, fallback_slices=fallback_Ts or None)

# ##########################################################################
# 5. Portfolio Greeks + scenarios
# ##########################################################################
T_nearest = max(sl.expiry_time for sl in fs.fitted_slices)

# Round spot to nearest 5 for a sensible strike
rounded_spot = int(round(spot / 5.0) * 5.0)

positions = [
    (OptionContract(symbol=symbol, option_type=OptionType.CALL,
                    strike=float(rounded_spot), expiry_date=date.today()),
     T_nearest, 1.0),
    (OptionContract(symbol=symbol, option_type=OptionType.PUT,
                    strike=float(rounded_spot * 0.95), expiry_date=date.today()),
     T_nearest, -0.5),
]
# ##########################################################################
# 6. Save 7 PNGs
# ##########################################################################
print("Saving 7 plots...")

# 1. 3D surface ribbons
from arbfree_vol.viz.surface import plot_surface
fig = plot_surface(list(reports["SVI"].fitted_slices))
fig.savefig(str(_OUT / "yfinance_demo_surface.png"), dpi=150)
print("  saved: yfinance_demo_surface.png")

# 2. Smile model comparison (new)
from arbfree_vol.viz.smiles import plot_smile_model_comparison
fig = plot_smile_model_comparison(surface, reports, symbol=symbol)
fig.savefig(str(_OUT / "yfinance_demo_smiles_comparison.png"), dpi=150)
print("  saved: yfinance_demo_smiles_comparison.png")

# 3. Model fit comparison bar chart (new)
from arbfree_vol.viz.comparison import plot_model_comparison
fig = plot_model_comparison(reports, symbol=symbol)
fig.savefig(str(_OUT / "yfinance_demo_model_comparison.png"), dpi=150)
print("  saved: yfinance_demo_model_comparison.png")

# 4. IV heatmap from FittedSurface (new)
from arbfree_vol.viz.surface import plot_iv_heatmap
fig = plot_iv_heatmap(fs, symbol=symbol, fallback_slices=fallback_Ts)
fig.savefig(str(_OUT / "yfinance_demo_iv_heatmap.png"), dpi=150)
print("  saved: yfinance_demo_iv_heatmap.png")

# 5. Dupire heatmap (new)
from arbfree_vol.viz.local_vol import plot_dupire_heatmap
fig = plot_dupire_heatmap(lv, symbol=symbol, fallback_slices=fallback_Ts)
fig.savefig(str(_OUT / "yfinance_demo_dupire.png"), dpi=150)
print("  saved: yfinance_demo_dupire.png")

# 6. Greeks heatmap (new)
from arbfree_vol.viz.risk import plot_greeks_heatmap
fig = plot_greeks_heatmap(fs, strikes, maturities, symbol=symbol,
                          fallback_slices=fallback_Ts)
fig.savefig(str(_OUT / "yfinance_demo_greeks.png"), dpi=150)
print("  saved: yfinance_demo_greeks.png")

# 7. Repair comparison (existing)
from arbfree_vol.viz.comparison import plot_comparison
fig = plot_comparison(reports["SVI"], reports["SVI"])
fig.savefig(str(_OUT / "yfinance_demo_repair.png"), dpi=150)
print("  saved: yfinance_demo_repair.png")

print(f"Done. 7 plots saved to {_OUT / 'yfinance_demo_*.png'}")
print(f"Run with:  python {Path(__file__).name}")
