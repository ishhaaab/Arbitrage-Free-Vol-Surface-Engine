"""Demo: yfinance -> repair -> mispricing backtest (single-cohort, frozen-vol hedge).

Limitation
----------
This is a **single-snapshot** backtest.  All trades are entered on the same
date using a single fitted surface.  The delta hedge uses the **frozen-vol**
convention (entry market IV held constant).  This is NOT a rolling daily-refit
backtest — it is a cross-sectional signal-research tool.

Output
------
4 PNGs saved to ``examples/backtest_demo_*.png``:
  - pnl_dist, cumulative, mispricing_scatter, metrics
"""

import matplotlib

matplotlib.use("Agg")  # save to files, no GUI

# yfinance is a soft dependency; gracefully exit if missing.
try:
    import yfinance as yf  # noqa: F401
except ImportError:
    print("yfinance is required.  Install with:  pip install yfinance")
    raise SystemExit(1)

from datetime import date

from arbfree_vol.ingestion.yfinance import fetch_chain
from arbfree_vol.repair.engine import repair
from arbfree_vol.surface.interpolate import build_fitted_surface
from arbfree_vol.backtest.engine import run_backtest

# ##########################################################################
# 1. Fetch + clean
# ##########################################################################
symbol = "SPY"
snapshot = date.today()
print(f"[{snapshot}] Fetching {symbol} chain for backtest...")
surface, rejected = fetch_chain(symbol, max_expiries=20, min_T_years=14.0 / 365.0)

T_count = len(surface.slices)
Q_count = sum(len(s.quotes) for s in surface.slices)
print(f"  Raw quotes fetched: {Q_count + len(rejected)}")
print(f"  Rejected by cleaning: {len(rejected)} "
      f"({len(rejected) / max(Q_count + len(rejected), 1) * 100:.1f}%)")
print(f"  Kept quotes: {Q_count}")
print(f"  Expiries: {T_count}")
print(f"  Spot={surface.spot:.2f}, r={surface.risk_free:.4f}, "
      f"q={surface.div_yield:.4f}")

# ##########################################################################
# 2. Repair (SVI default)
# ##########################################################################
print("Repairing with SVI...")
report = repair(surface)
n_before = report.metrics.n_violations_before
n_after = report.metrics.n_violations_after
n_rej = report.metrics.n_rejected
n_fitted = len(report.fitted_slices)
avg_rmse = (
    sum(fs.rmse for fs in report.fitted_slices) / n_fitted
    if n_fitted > 0 else 0.0
)
print(f"  Violations: {n_before} -> {n_after}, rejected={n_rej}, "
      f"fitted={n_fitted}, avg_RMSE={avg_rmse:.4f}")

# ##########################################################################
# 3. Build FittedSurface
# ##########################################################################
print("Building fitted surface...")
fs = build_fitted_surface(report)

# ##########################################################################
# 4. Run backtest
# ##########################################################################
print("Running mispricing backtest (single-cohort, frozen-vol delta hedge)...")
result = run_backtest(
    surface=surface,
    fs=fs,
    symbol=symbol,
    snapshot_date=snapshot,
    threshold=0.01,
)

print(f"  Trades entered: {result.n_trades}")
print(f"  Hit rate:       {result.hit_rate:.2%}")
print(f"  Total P&L:      ${result.total_pnl:.2f}")
print(f"  Mean P&L:       ${result.mean_pnl:.3f}")
print(f"  Std P&L:        ${result.std_pnl:.3f}")
print(f"  Sharpe:         {result.sharpe:.3f}")
print(f"  Max drawdown:   ${result.max_drawdown:.2f}")
print(f"  P&L P5/P50/P95: ${result.pnl_p5:.2f} / ${result.pnl_p50:.2f} / "
      f"${result.pnl_p95:.2f}")

if result.n_trades > 0:
    n_long = sum(1 for t in result.trades if t.signal.side == 1)
    n_short = sum(1 for t in result.trades if t.signal.side == -1)
    print(f"  Long trades:  {n_long}")
    print(f"  Short trades: {n_short}")

# ##########################################################################
# 5. Save 4 PNGs
# ##########################################################################
print("Saving 4 backtest plots...")

from arbfree_vol.viz.backtest import (
    plot_pnl_distribution,
    plot_cumulative_pnl,
    plot_mispricing_vs_pnl,
    plot_backtest_metrics,
)

fig = plot_pnl_distribution(result, symbol=symbol)
fig.savefig("examples/backtest_demo_pnl_dist.png", dpi=150)
print("  saved: backtest_demo_pnl_dist.png")

fig = plot_cumulative_pnl(result, symbol=symbol)
fig.savefig("examples/backtest_demo_cumulative.png", dpi=150)
print("  saved: backtest_demo_cumulative.png")

fig = plot_mispricing_vs_pnl(result, symbol=symbol)
fig.savefig("examples/backtest_demo_mispricing_scatter.png", dpi=150)
print("  saved: backtest_demo_mispricing_scatter.png")

fig = plot_backtest_metrics(result, symbol=symbol)
fig.savefig("examples/backtest_demo_metrics.png", dpi=150)
print("  saved: backtest_demo_metrics.png")

print("Done. 4 plots saved to examples/backtest_demo_*.png")
print()
print("Note: This is a single-snapshot / frozen-vol-hedge backtest — not a rolling")
print("daily-refit backtest. All trades share the same entry date and surface fit.")
