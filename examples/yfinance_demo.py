"""Demo: yfinance → clean → VolSurface → repair → visualize.

Uses the ``ingestion.yfinance`` module which sources real risk-free
rates (^IRX) and dividend yields, fetches mid prices, and applies
the cleaning layer before building the surface.
"""

import sys; sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg")

from arbfree_vol.ingestion.yfinance import fetch_chain
from arbfree_vol.repair.engine import repair
from arbfree_vol.viz.surface import plot_surface, plot_heatmap_2d
from arbfree_vol.viz.smiles import plot_smiles, plot_smiles_heatmap
from arbfree_vol.viz.comparison import plot_comparison

# --- 1. Fetch + clean ---
symbol = "SPY"
print(f"Fetching {symbol} chain (mid prices, real r/q, with cleaning).")
surface, rejected = fetch_chain(symbol, max_expiries=20, min_T_years=7.0 / 365.0)

T_count = len(surface.slices)
Q_count = sum(len(s.quotes) for s in surface.slices)
print(f"  Raw quotes fetched: {Q_count + len(rejected)}")
print(f"  Rejected by cleaning: {len(rejected)} ({len(rejected) / max(Q_count + len(rejected), 1) * 100:.1f}%)")
print(f"  Kept quotes: {Q_count}")
print(f"  Expiries: {T_count}")
print(f"  Spot={surface.spot:.2f}, r={surface.risk_free:.4f}, q={surface.div_yield:.4f}")
if rejected:
    rules = {r.rule.value for r in rejected}
    print(f"  Rejection reasons: {rules}")

# --- 2. Repair ---
print("Repairing.")
report = repair(surface)

print(f"  Violations before: {report.metrics.n_violations_before}")
print(f"  Violations after:  {report.metrics.n_violations_after}")
print(f"  Quotes rejected by repair: {report.metrics.n_rejected}")
print(f"  Slices fitted:     {report.metrics.n_slices_fitted}")
print(f"  Total rejection rate: {report.metrics.rejection_rate * 100:.1f}%")
for v in report.remaining_violations.violations:
    print(f"  Remaining: {v.kind.value} — {v.detail[:80]}")

# --- 3. Save plots ---
print("Saving plots")
fig1 = plot_surface(list(report.fitted_slices))
fig1.savefig("examples/yfinance_demo_surface.png", dpi=150)

fig2 = plot_smiles(surface, list(report.fitted_slices))
fig2.savefig("examples/yfinance_demo_smiles.png", dpi=150)

fig3 = plot_comparison(report, report)
fig3.savefig("examples/yfinance_demo_comparison.png", dpi=150)

fig4 = plot_heatmap_2d(list(report.fitted_slices), symbol=symbol)
fig4.savefig("examples/yfinance_demo_heatmap.png", dpi=150)

fig5 = plot_smiles_heatmap(list(report.fitted_slices), symbol=symbol)
fig5.savefig("examples/yfinance_demo_smiles_heatmap.png", dpi=150)

print("Done, plots saved to examples/yfinance_demo_*.png")
