"""eSSVI surface demo — where eSSVI shines: long-dated, less skewed data.

Fits an eSSVI surface to long-dated SPY options (T > 60 days) and
saves the smoothed surface plot for comparison against the raw SVI
surface from ``yfinance_demo.py``.
"""

import sys; sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg")

from arbfree_vol.ingestion.yfinance import fetch_chain
from arbfree_vol.repair.engine import repair
from arbfree_vol.viz.surface import plot_surface
from arbfree_vol.viz.smiles import plot_smiles

symbol = "SPY"
print(f"Fetching {symbol} long-dated chains (T > 60 days)")

surface, _ = fetch_chain(symbol, max_expiries=12, min_T_years=60.0 / 365.0)

print(f"Loaded {sum(len(s.quotes) for s in surface.slices)} quotes "
      f"across {len(surface.slices)} expiries")

# Raw SVI fit
print("Fitting raw SVI...")
r_svi = repair(surface, use_ssvi=False)
print(f"  Remaining: {r_svi.metrics.n_violations_after} violations")

# eSSVI fit
print("Fitting eSSVI...")
r_essvi = repair(surface, use_ssvi=True)
print(f"  Remaining: {r_essvi.metrics.n_violations_after} violations")

# Save surface plots side by side
fig_svi = plot_surface(list(r_svi.fitted_slices))
fig_svi.savefig("examples/essvi_demo_raw_svi_surface.png", dpi=150)
print("Saved examples/essvi_demo_raw_svi_surface.png")

fig_essvi = plot_surface(list(r_essvi.fitted_slices))
fig_essvi.savefig("examples/essvi_demo_essvi_surface.png", dpi=150)
print("Saved examples/essvi_demo_essvi_surface.png")

fig_smiles_svi = plot_smiles(surface, list(r_svi.fitted_slices))
fig_smiles_svi.savefig("examples/essvi_demo_raw_svi_smiles.png", dpi=150)

fig_smiles_essvi = plot_smiles(surface, list(r_essvi.fitted_slices))
fig_smiles_essvi.savefig("examples/essvi_demo_essvi_smiles.png", dpi=150)
print("Saved smiles for both fits")
