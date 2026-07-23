"""Demo: yfinance -> repair with SVI/eSSVI/SABR -> fitted surface -> Greeks -> Dupire -> 8 plots.

Uses the ``ingestion.yfinance`` module which sources real risk-free
rates (^IRX) and dividend yields, fetches mid prices, and applies
the cleaning layer before building the surface.
"""

import matplotlib
matplotlib.use("Agg")  # save to files, no GUI

# yfinance is a soft dependency; gracefully exit if missing.
try:
    import yfinance as yf  # noqa: F401
except ImportError:
    print("yfinance is required.  Install with:  pip install yfinance")
    raise SystemExit(1)

from collections import Counter
from datetime import date

import numpy as np

from arbfree_vol.ingestion.yfinance import fetch_chain
from arbfree_vol.repair.engine import repair
from arbfree_vol.surface.interpolate import build_fitted_surface
from arbfree_vol.pricing.local_vol import dupire
from arbfree_vol.surface.risk import spot_bump_analysis
from arbfree_vol.models.option import OptionContract, OptionType

# ##########################################################################
# 1. Fetch + clean
# ##########################################################################
symbol = "SPY"
print(f"Fetching {symbol} chain (mid prices, real r/q, with cleaning)...")
surface, rejected = fetch_chain(symbol, max_expiries=20, min_T_years=7.0 / 365.0)

T_count = len(surface.slices)
Q_count = sum(len(s.quotes) for s in surface.slices)
print(f"  Raw quotes fetched: {Q_count + len(rejected)}")
print(f"  Rejected by cleaning: {len(rejected)} "
      f"({len(rejected) / max(Q_count + len(rejected), 1) * 100:.1f}%)")
print(f"  Kept quotes: {Q_count}")
print(f"  Expiries: {T_count}")
print(f"  Spot={surface.spot:.2f}, r={surface.risk_free:.4f}, "
      f"q={surface.div_yield:.4f}")
if rejected:
    rule_counts = Counter(r.rule.value for r in rejected)
    print(f"  Rejection breakdown (first rule hit per quote):")
    for rule, count in rule_counts.most_common():
        print(f"    {rule}: {count} ({count / len(rejected) * 100:.1f}%)")

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

# ##########################################################################
# 3. Build FittedSurface (use SVI report)
# ##########################################################################
fs = build_fitted_surface(reports["SVI"])

# ##########################################################################
# 4. Dupire local vol
# ##########################################################################
spot = surface.spot
strikes = list(np.linspace(spot * 0.85, spot * 1.15, 20))

# Maturities: base from surface, then pad to >= 3 points.
mat_base = sorted(sl.expiry_time for sl in surface.slices)
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
lv = dupire(fs, strikes, maturities)

# ##########################################################################
# 5. Portfolio Greeks + scenarios
# ##########################################################################
T_nearest = max(sl.expiry_time for sl in surface.slices)

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
scenarios = spot_bump_analysis(
    fs, positions,
    bumps=[-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10],
)

# ##########################################################################
# 6. Save 8 PNGs
# ##########################################################################
print("Saving 8 plots...")

# 1. 3D surface ribbons
from arbfree_vol.viz.surface import plot_surface
fig = plot_surface(list(reports["SVI"].fitted_slices))
fig.savefig("examples/yfinance_demo_surface.png", dpi=150)
print("  saved: yfinance_demo_surface.png")

# 2. Smile model comparison (new)
from arbfree_vol.viz.smiles import plot_smile_model_comparison
fig = plot_smile_model_comparison(surface, reports, symbol=symbol)
fig.savefig("examples/yfinance_demo_smiles_comparison.png", dpi=150)
print("  saved: yfinance_demo_smiles_comparison.png")

# 3. Model fit comparison bar chart (new)
from arbfree_vol.viz.comparison import plot_model_comparison
fig = plot_model_comparison(reports, symbol=symbol)
fig.savefig("examples/yfinance_demo_model_comparison.png", dpi=150)
print("  saved: yfinance_demo_model_comparison.png")

# 4. IV heatmap from FittedSurface (new)
from arbfree_vol.viz.surface import plot_iv_heatmap
fig = plot_iv_heatmap(fs, symbol=symbol)
fig.savefig("examples/yfinance_demo_iv_heatmap.png", dpi=150)
print("  saved: yfinance_demo_iv_heatmap.png")

# 5. Dupire heatmap (new)
from arbfree_vol.viz.local_vol import plot_dupire_heatmap
fig = plot_dupire_heatmap(lv, symbol=symbol)
fig.savefig("examples/yfinance_demo_dupire.png", dpi=150)
print("  saved: yfinance_demo_dupire.png")

# 6. Greeks heatmap (new)
from arbfree_vol.viz.risk import plot_greeks_heatmap
fig = plot_greeks_heatmap(fs, strikes, maturities, symbol=symbol)
fig.savefig("examples/yfinance_demo_greeks.png", dpi=150)
print("  saved: yfinance_demo_greeks.png")

# 7. Scenario payoff bar chart (new)
from arbfree_vol.viz.risk import plot_scenario_payoff
fig = plot_scenario_payoff(scenarios, symbol=symbol)
fig.savefig("examples/yfinance_demo_scenario.png", dpi=150)
print("  saved: yfinance_demo_scenario.png")

# 8. Repair comparison (existing)
from arbfree_vol.viz.comparison import plot_comparison
fig = plot_comparison(reports["SVI"], reports["SVI"])
fig.savefig("examples/yfinance_demo_repair.png", dpi=150)
print("  saved: yfinance_demo_repair.png")

print("Done. 8 plots saved to examples/yfinance_demo_*.png")
print("Run with:  python examples/yfinance_demo.py")
