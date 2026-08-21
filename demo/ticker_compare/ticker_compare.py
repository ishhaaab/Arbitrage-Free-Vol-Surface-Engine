"""Cross-ticker volatility-surface comparison.

Fetches live option chains for several tickers, repairs each with SVI,
eSSVI, and SABR, and produces three comparison plots:

  1. ATM term structure          — one line per ticker
  2. ~30-day smile overlay       — one line per ticker
  3. Grouped bar of median RMSE   — per model per ticker

Live mode needs yfinance. ``--offline`` falls back to a deterministic
synthetic surface per ticker (no network).

Usage::

    python demo/ticker_compare/ticker_compare.py                  # SPY,QQQ,IWM (live)
    python demo/ticker_compare/ticker_compare.py --offline        # synthetic (no network)
    python demo/ticker_compare/ticker_compare.py --tickers SPY,AAPL,MSFT
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save to files, no GUI window

import matplotlib.pyplot as plt
import numpy as np

_OUT = Path(__file__).parent

parser = argparse.ArgumentParser(
    description="Cross-ticker vol-surface comparison (SVI / eSSVI / SABR)",
)
parser.add_argument(
    "--tickers",
    default="SPY,QQQ,IWM",
    help="Comma-separated ticker symbols, e.g. SPY,QQQ,IWM",
)
parser.add_argument(
    "--offline",
    action="store_true",
    help="Use deterministic synthetic surfaces instead of live data.",
)
parser.add_argument(
    "--max-expiries", type=int, default=12,
    help="Number of expiries per ticker (live mode only).",
)
args = parser.parse_args()
tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
OFFLINE = args.offline
MAX_EXPIRIES = args.max_expiries

# Distinct colours for up to 5 tickers.
_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
_MODEL_LABELS = ["SVI", "eSSVI", "SABR"]
_TARGET_T = 30.0 / 365.25  # ~0.0822 years for the smile overlay


# ---------------------------------------------------------------------------
# 1. Data: live yfinance OR synthetic fallback
# ---------------------------------------------------------------------------
def _fetch_ticker(ticker: str):
    """Return (surface, reports) for one ticker.

    ``reports`` is a dict of label -> RepairReport for SVI/eSSVI/SABR.
    Raises on failure (caller decides whether to skip).
    """
    if OFFLINE:
        surface = _synthetic_surface(ticker)
    else:
        from arbfree_vol.ingestion.yahoo import fetch_chain
        surface, _, _ = fetch_chain(
            ticker, max_expiries=MAX_EXPIRIES, min_T_years=7.0 / 365.0,
            use_fred_curve=True, day_count="ACT/365F", calendar="USNYSE",
        )

    from arbfree_vol.repair.engine import repair
    reports = {}
    for label, kw in [
        ("SVI",   {"use_ssvi": False, "use_sabr": False}),
        ("eSSVI", {"use_ssvi": True,  "use_sabr": False}),
        ("SABR",  {"use_ssvi": False, "use_sabr": True}),
    ]:
        reports[label] = repair(surface, **kw)
    return surface, reports


def _synthetic_surface(ticker: str):
    """Deterministic synthetic surface per ticker (no network).

    Same 3-expiry SVI shape for every ticker, but with a ticker-seeded
    smile so the plots differ visibly across symbols.
    """
    from math import sqrt

    from arbfree_vol.models.option import OptionType
    from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
    from arbfree_vol.svi.model import SVIParams, svi_total_variance
    from arbfree_vol.pricing.black_scholes import price_floats

    seed = sum(ord(c) for c in ticker)
    spot, r, q = 100.0, 0.05, 0.01
    exps = [0.25, 0.5, 1.0]
    params = {
        T: SVIParams(
            a=0.008 + 0.002 * ((seed + int(T * 100)) % 5) / 5.0,
            b=0.30 + 0.05 * ((seed + int(T * 100)) % 3) / 3.0,
            rho=-0.25 - 0.1 * ((seed + int(T * 100)) % 4) / 4.0,
            m=0.0,
            sigma=0.10 + 0.03 * ((seed + int(T * 100)) % 6) / 6.0,
        )
        for T in exps
    }

    def bs_price(otype, K, sigma, tt):
        return price_floats(spot, K, tt, r, q, sigma,
                            is_call=(otype == OptionType.CALL))

    slices = []
    for T in exps:
        ks = np.linspace(-0.30, 0.30, 9)
        quotes = []
        for k in ks:
            w = svi_total_variance(float(k), params[T].a, params[T].b,
                                   params[T].rho, params[T].m, params[T].sigma)
            sigma = sqrt(w / T)
            for otype in (OptionType.CALL, OptionType.PUT):
                K = float(spot * np.exp(k))
                price = bs_price(otype, K, sigma, T)
                quotes.append(Quote(strike=K, option_type=otype, price=price))
        slices.append(ExpirySlice(expiry_time=T, quotes=quotes))
    return VolSurface(spot=spot, risk_free=r, div_yield=q, slices=slices)


# ---------------------------------------------------------------------------
# 2. Build FittedSurface for the SVI report (query layer)
# ---------------------------------------------------------------------------
def _build_fs(reports):
    from arbfree_vol.surface.interpolate import build_fitted_surface
    return build_fitted_surface(reports["SVI"])


# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------
def main() -> None:
    print()
    print("=" * 62)
    print("   Cross-ticker vol-surface comparison")
    print("   " + ("LIVE data" if not OFFLINE else "SYNTHETIC data (no network)"))
    print("=" * 62)

    results: dict[str, dict] = {}
    for ticker in tickers:
        print(f"\n  --- {ticker} ---")
        try:
            surface, reports = _fetch_ticker(ticker)
            fs = _build_fs(reports)
            results[ticker] = {
                "surface": surface,
                "fs": fs,
                "reports": reports,
                "n_kept": len(surface.slices),
            }
            for label in _MODEL_LABELS:
                r = reports[label]
                n = len(r.fitted_slices)
                med = (
                    float(np.median([s.rmse for s in r.fitted_slices]))
                    if n > 0 else 0.0
                )
                print(f"    {label:6s}: fitted={n}, median_RMSE={med:.4f}")
        except Exception as exc:
            print(f"    SKIPPED {ticker}: {exc}")

    if not results:
        print("\nNo tickers succeeded. Nothing to plot.")
        raise SystemExit(0)

    # Plot 1 — ATM term structure
    from arbfree_vol.surface.interpolate import iv_at
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
        ax1.plot(ts, ivs, marker="o", label=ticker,
                 color=_COLORS[idx % len(_COLORS)])
    ax1.set_xlabel("Expiry (days)")
    ax1.set_ylabel("ATM implied vol")
    ax1.set_title("ATM term structure")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(str(_OUT / "ticker_compare_atm_term_structure.png"), dpi=150)
    plt.close(fig1)
    print("\nSaved: ticker_compare_atm_term_structure.png")

    # Plot 2 — ~30-day smile overlay
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    for idx, ticker in enumerate(results):
        fs = results[ticker]["fs"]
        slices = fs.fitted_slices
        if not slices:
            continue
        nearest = min(slices, key=lambda sl: abs(sl.expiry_time - _TARGET_T))
        T = nearest.expiry_time
        F = nearest.forward_price
        strikes = np.linspace(0.85 * F, 1.15 * F, 30)
        smile_ivs, valid_strikes = [], []
        for K in strikes:
            try:
                smile_ivs.append(iv_at(fs, K, T))
                valid_strikes.append(K)
            except (ValueError, ZeroDivisionError):
                continue
        if valid_strikes:
            ax2.plot(np.array(valid_strikes) / F, smile_ivs, marker=".",
                     label=f"{ticker} (T={T*365.25:.0f}d)",
                     color=_COLORS[idx % len(_COLORS)])
    ax2.set_xlabel("Strike / Forward")
    ax2.set_ylabel("Implied vol")
    ax2.set_title("Smile comparison (~30-day expiry)")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(str(_OUT / "ticker_compare_smiles.png"), dpi=150)
    plt.close(fig2)
    print("Saved: ticker_compare_smiles.png")

    # Plot 3 — RMSE grouped bar (median across fitted slices, so a couple
    # of noisy expiries don't dominate the comparison).
    tickers_ok = list(results.keys())
    x = np.arange(len(tickers_ok))
    bar_width = 0.25
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for m_idx, label in enumerate(_MODEL_LABELS):
        rmse_vals = []
        for ticker in tickers_ok:
            r = results[ticker]["reports"][label]
            n = len(r.fitted_slices)
            med = (
                float(np.median([s.rmse for s in r.fitted_slices]))
                if n > 0 else 0.0
            )
            rmse_vals.append(med)
        ax3.bar(x + m_idx * bar_width, rmse_vals, bar_width, label=label)
    ax3.set_xlabel("Ticker")
    ax3.set_ylabel("Median RMSE")
    ax3.set_title("Model fit quality (median RMSE across expiries)")
    ax3.set_xticks(x + bar_width)
    ax3.set_xticklabels(tickers_ok)
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(str(_OUT / "ticker_compare_rmse.png"), dpi=150)
    plt.close(fig3)
    print("Saved: ticker_compare_rmse.png")

    # Summary table
    print(f"\n{'Ticker':<8} {'Kept':>5} {'Fitted':>7}", end="")
    for label in _MODEL_LABELS:
        print(f"  {label:>8}", end="")
    print()
    print("-" * (8 + 5 + 7 + 3 * 10 + 2))
    for ticker in tickers_ok:
        reports = results[ticker]["reports"]
        n_kept = results[ticker]["n_kept"]
        n_fitted = len(reports["SVI"].fitted_slices)
        print(f"{ticker:<8} {n_kept:>5} {n_fitted:>7}", end="")
        for label in _MODEL_LABELS:
            r = reports[label]
            n = len(r.fitted_slices)
            med = (
                float(np.median([s.rmse for s in r.fitted_slices]))
                if n > 0 else 0.0
            )
            print(f"  {med:>8.4f}", end="")
        print()

    failed = [t for t in tickers if t not in results]
    if failed:
        print(f"\nSkipped (fetch/fit failed): {', '.join(failed)}")
    print(f"\n3 plots saved to {_OUT / 'ticker_compare_*.png'}")


if __name__ == "__main__":
    main()
