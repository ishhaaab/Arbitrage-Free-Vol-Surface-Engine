"""Cross-ticker volatility-surface comparison demo.

Fetches live option chains for multiple tickers, repairs each with
SVI, eSSVI, and SABR, and produces three comparison plots:

  1. ATM term structure  (one line per ticker)
  2. ~30-day smile overlay  (one line per ticker)
  3. Grouped bar chart of average RMSE per model per ticker

Needs network access and yfinance.  Takes 2-5 minutes for 3 tickers.

Usage::

    python demo/ticker_compare/ticker_compare.py                    # default: SPY,QQQ,IWM
    python demo/ticker_compare/ticker_compare.py --tickers SPY,AAPL # custom list
"""

import matplotlib
matplotlib.use("Agg")

# yfinance is a soft dependency; gracefully exit if missing.
try:
    import yfinance as yf  # noqa: F401
except ImportError:
    print("yfinance is required.  Install with:  pip install yfinance")
    raise SystemExit(1)

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from arbfree_vol.ingestion.yahoo import fetch_chain
from arbfree_vol.repair.engine import repair
from arbfree_vol.surface.interpolate import build_fitted_surface, iv_at

# ##########################################################################
# 0. Parse CLI args
# ##########################################################################
parser = argparse.ArgumentParser(
    description="Cross-ticker vol-surface comparison (SVI / eSSVI / SABR)",
)
parser.add_argument(
    "--tickers",
    default="SPY,QQQ,IWM",
    help="Comma-separated ticker symbols, e.g. SPY,QQQ,IWM",
)
args = parser.parse_args()
tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
_OUT = Path(__file__).parent

# Distinct colours for up to 5 tickers.
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# ##########################################################################
# 1. Fetch, repair, build surfaces for each ticker
# ##########################################################################
# results[ticker] = {
#     "fs": FittedSurface,
#     "reports": {"SVI": RepairReport, "eSSVI": ..., "SABR": ...},
#     "n_kept": int,
# }
results: dict[str, dict] = {}
model_labels = ["SVI", "eSSVI", "SABR"]

for ticker in tickers:
    print(f"\n{'='*60}")
    print(f"  {ticker}")
    print(f"{'='*60}")
    try:
        surface, rejected, quality_drops = fetch_chain(ticker, max_expiries=20, min_T_years=7.0 / 365.0)
        n_kept = len(surface.slices)
        print(f"  Expiries kept: {n_kept}  (rejected {len(rejected)} quotes)")

        reports: dict[str, object] = {}
        for label, kw in [
            ("SVI",  {"use_ssvi": False, "use_sabr": False}),
            ("eSSVI", {"use_ssvi": True,  "use_sabr": False}),
            ("SABR",  {"use_ssvi": False, "use_sabr": True}),
        ]:
            r = repair(surface, **kw)
            reports[label] = r
            n_fitted = len(r.fitted_slices)
            avg_rmse = (
                sum(fs.rmse for fs in r.fitted_slices) / n_fitted
                if n_fitted > 0 else 0.0
            )
            print(f"  {label:6s}: fitted={n_fitted}, avg_RMSE={avg_rmse:.4f}")

        fs = build_fitted_surface(reports["SVI"])
        results[ticker] = {"fs": fs, "reports": reports, "n_kept": n_kept}

    except Exception as exc:
        print(f"  SKIPPED {ticker}: {exc}")

if not results:
    print("\nNo tickers succeeded. Nothing to plot.")
    raise SystemExit(0)

# ##########################################################################
# 2. Plot 1 — ATM term structure
# ##########################################################################
fig1, ax1 = plt.subplots(figsize=(8, 5))
for idx, ticker in enumerate(results):
    fs = results[ticker]["fs"]
    ts, ivs = [], []
    for sl in fs.fitted_slices:
        F = sl.forward_price
        T = sl.expiry_time
        try:
            iv = iv_at(fs, F, T)
            ts.append(T * 365.25)
            ivs.append(iv)
        except (ValueError, ZeroDivisionError):
            continue
    color = _COLORS[idx % len(_COLORS)]
    ax1.plot(ts, ivs, marker="o", label=ticker, color=color)

ax1.set_xlabel("Expiry (days)")
ax1.set_ylabel("ATM implied vol")
ax1.set_title("ATM term structure")
ax1.legend()
fig1.tight_layout()
fig1.savefig(str(_OUT / "ticker_compare_atm_term_structure.png"), dpi=150)
plt.close(fig1)
print(f"\nSaved: ticker_compare_atm_term_structure.png")

# ##########################################################################
# 3. Plot 2 — ~30-day smile overlay
# ##########################################################################
_TARGET_T = 30.0 / 365.25  # ~0.0822 years

fig2, ax2 = plt.subplots(figsize=(8, 5))
for idx, ticker in enumerate(results):
    fs = results[ticker]["fs"]
    slices = fs.fitted_slices
    if not slices:
        continue
    # Pick the slice closest to 30 days.
    nearest = min(slices, key=lambda sl: abs(sl.expiry_time - _TARGET_T))
    T = nearest.expiry_time
    F = nearest.forward_price
    strikes = np.linspace(0.85 * F, 1.15 * F, 30)
    smile_ivs = []
    valid_strikes = []
    for K in strikes:
        try:
            iv = iv_at(fs, K, T)
            smile_ivs.append(iv)
            valid_strikes.append(K)
        except (ValueError, ZeroDivisionError):
            continue
    if valid_strikes:
        color = _COLORS[idx % len(_COLORS)]
        moneyness = np.array(valid_strikes) / F
        ax2.plot(moneyness, smile_ivs, marker=".", label=f"{ticker} (T={T*365.25:.0f}d)", color=color)

ax2.set_xlabel("Strike / Forward")
ax2.set_ylabel("Implied vol")
ax2.set_title("Smile comparison (~30-day expiry)")
ax2.legend()
fig2.tight_layout()
fig2.savefig(str(_OUT / "ticker_compare_smiles.png"), dpi=150)
plt.close(fig2)
print(f"Saved: ticker_compare_smiles.png")

# ##########################################################################
# 4. Plot 3 — RMSE grouped bar chart
# ##########################################################################
tickers_ok = list(results.keys())
x = np.arange(len(tickers_ok))
bar_width = 0.25

fig3, ax3 = plt.subplots(figsize=(8, 5))
for m_idx, label in enumerate(model_labels):
    rmse_vals = []
    for ticker in tickers_ok:
        r = results[ticker]["reports"][label]
        n_fitted = len(r.fitted_slices)
        avg = (
            sum(fs.rmse for fs in r.fitted_slices) / n_fitted
            if n_fitted > 0 else 0.0
        )
        rmse_vals.append(avg)
    ax3.bar(x + m_idx * bar_width, rmse_vals, bar_width, label=label)

ax3.set_xlabel("Ticker")
ax3.set_ylabel("Average RMSE")
ax3.set_title("Model fit quality (avg RMSE across expiries)")
ax3.set_xticks(x + bar_width)
ax3.set_xticklabels(tickers_ok)
ax3.legend()
fig3.tight_layout()
fig3.savefig(str(_OUT / "ticker_compare_rmse.png"), dpi=150)
plt.close(fig3)
print(f"Saved: ticker_compare_rmse.png")

# ##########################################################################
# 5. Summary table
# ##########################################################################
print(f"\n{'Ticker':<8} {'Kept':>5} {'Fitted':>7}", end="")
for label in model_labels:
    print(f"  {label:>8}", end="")
print()
print("-" * (8 + 5 + 7 + 3 * 10 + 2))
for ticker in tickers_ok:
    fs = results[ticker]["fs"]
    reports = results[ticker]["reports"]
    n_kept = results[ticker]["n_kept"]
    n_fitted_svi = len(reports["SVI"].fitted_slices)
    print(f"{ticker:<8} {n_kept:>5} {n_fitted_svi:>7}", end="")
    for label in model_labels:
        r = reports[label]
        n = len(r.fitted_slices)
        avg = sum(s.rmse for s in r.fitted_slices) / n if n > 0 else 0.0
        print(f"  {avg:>8.4f}", end="")
    print()

failed = [t for t in tickers if t not in results]
if failed:
    print(f"\nFailed tickers: {', '.join(failed)}")
