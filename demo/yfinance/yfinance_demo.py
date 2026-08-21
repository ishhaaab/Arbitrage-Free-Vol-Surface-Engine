"""Modern yfinance pipeline demo — live or synthetic, streamlined core.

Shows the current (2026-08) end-to-end pipeline on real option data:

    fetch_chain  ->  repair (SVI / eSSVI / SABR)  ->  FittedSurface
                ->  Dupire local vol  ->  Greeks  ->  plots

Demonstrates the modern rate/calendar seams that the old demo predated:

* ``fetch_chain(..., use_fred_curve=True)`` — risk-free rates from the
  FRED treasury/SOFR ``YieldTermStructure`` (per-slice r(T) threaded);
  when FRED is unavailable it falls back to a flat 5% curve.
* ``day_count=`` and ``calendar=`` — ACT/365F + USNYSE roll.
* eSSVI fallback slices (H&M Prop 3.1 hard-constrained fit failed) are
  grayed out in the heatmaps and flagged in the console output.

Usage::

    python demo/yfinance/yfinance_demo.py                # live SPY
    python demo/yfinance/yfinance_demo.py --symbol QQQ    # live QQQ
    python demo/yfinance/yfinance_demo.py --offline       # synthetic (no network)

Outputs 6 PNGs into ``demo/yfinance/``:

    yfinance_demo_surface.png       3-D surface ribbons (SVI fit)
    yfinance_demo_smiles.png        per-expiry smile: data + eSSVI fits
    yfinance_demo_iv_heatmap.png    IV heatmap over (strike, maturity)
    yfinance_demo_dupire.png        Dupire local-vol heatmap
    yfinance_demo_greeks.png        CALL Greeks heatmaps (delta/gamma/vega)
    yfinance_demo_violations.png    arbitrage violations detected pre-repair
"""

import argparse
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: save to files, no GUI window

_OUT = Path(__file__).parent

# ---------------------------------------------------------------------------
# 0. CLI
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="yfinance -> repair -> FittedSurface -> Dupire -> Greeks -> plots",
)
parser.add_argument("--symbol", default="SPY",
                    help="Ticker symbol (default: SPY) for live mode.")
parser.add_argument("--offline", action="store_true",
                    help="Use a deterministic synthetic surface instead of yfinance.")
parser.add_argument("--max-expiries", type=int, default=10,
                    help="Number of expiries to fetch (live mode only).")
parser.add_argument("--no-fred", action="store_true",
                    help="Disable the FRED rate curve (falls back to flat 5%).")
args = parser.parse_args()

SYMBOL = args.symbol
OFFLINE = args.offline
MAX_EXPIRIES = args.max_expiries
USE_FRED = not args.no_fred


def _sep(title: str) -> None:
    print()
    print("=" * 66)
    print(f"  {title}")
    print("=" * 66)


# ---------------------------------------------------------------------------
# 1. Data: live yfinance OR synthetic fallback
# ---------------------------------------------------------------------------
@dataclass
class DataBundle:
    surface: object                      # VolSurface
    rejected: list = field(default_factory=list)
    quality_drops: list = field(default_factory=list)
    source: str = "live"
    summary: str = ""


def _build_synthetic() -> DataBundle:
    """A deterministic 3-expiry SVI surface with injected arbitrage.

    Same shape as a fetched chain, but ``--offline`` produces identical
    plots every run.  A few quotes are deliberately mispriced so the
    repair stage has real violations to fix and the violations bar chart
    shows a real distribution (detector returns BUTTERFLY / PARITY /
    MONOTONICITY on this surface).
    """
    from math import sqrt

    from arbfree_vol.models.option import OptionType
    from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
    from arbfree_vol.svi.model import SVIParams, svi_total_variance

    spot, r, q = 100.0, 0.05, 0.01
    exps = [0.25, 0.5, 1.0]
    params = {
        0.25: SVIParams(a=0.010, b=0.32, rho=-0.35, m=0.0, sigma=0.11),
        0.5:  SVIParams(a=0.020, b=0.35, rho=-0.38, m=0.02, sigma=0.13),
        1.0:  SVIParams(a=0.035, b=0.40, rho=-0.42, m=0.04, sigma=0.16),
    }

    def bs_price(otype, K, sigma, tt):
        from arbfree_vol.pricing.black_scholes import price_floats
        return price_floats(spot, K, tt, r, q, sigma,
                            is_call=(otype == OptionType.CALL))

    slices = []
    for T in exps:
        ks = np.linspace(-0.30, 0.30, 9)
        quotes = []
        for i, k in enumerate(ks):
            w = svi_total_variance(float(k), params[T].a, params[T].b,
                                   params[T].rho, params[T].m, params[T].sigma)
            sigma = sqrt(w / T)
            for otype in (OptionType.CALL, OptionType.PUT):
                price = bs_price(otype, float(spot * np.exp(k)), sigma, T)
                # Inject a MIX of violations so the repair stage and the
                # violations bar chart show a real distribution (verified
                # via detect(): PARITY x3, BUTTERFLY x3, MONOTONICITY x2):
                #  * inflate the short-dated ATM put (breaks put-call parity)
                #  * discount a short-dated OTM call (breaks convexity:
                #    a butterfly spread becomes profitable)
                #  * discount a long-dated OTM call (breaks calendar
                #    monotonicity: a later expiry is priced below an earlier
                #    one at the same strike)
                if T == exps[0] and otype == OptionType.PUT and i == 4:
                    price *= 1.10
                elif T == exps[0] and otype == OptionType.CALL and i == 7:
                    price *= 0.40
                elif T == exps[2] and otype == OptionType.CALL and i == 6:
                    price *= 0.50
                quotes.append(Quote(
                    strike=float(spot * np.exp(k)), option_type=otype,
                    price=price,
                ))
        slices.append(ExpirySlice(expiry_time=T, quotes=quotes))
    surface = VolSurface(spot=spot, risk_free=r, div_yield=q, slices=slices)
    return DataBundle(
        surface=surface, source="synthetic",
        summary=(f"{sum(len(s.quotes) for s in slices)} quotes across "
                 f"{len(slices)} expiries (synthetic)"),
    )


def _fetch_live() -> DataBundle:
    """Fetch a live chain via the modern rate/calendar seams."""
    from arbfree_vol.ingestion.yahoo import fetch_chain

    surface, rejected, quality_drops = fetch_chain(
        SYMBOL,
        max_expiries=MAX_EXPIRIES,
        min_T_years=7.0 / 365.0,
        use_fred_curve=USE_FRED,
        day_count="ACT/365F",
        calendar="USNYSE",
    )
    return DataBundle(
        surface=surface, rejected=rejected, quality_drops=quality_drops,
        source="live",
        summary=(f"{sum(len(s.quotes) for s in surface.slices)} quotes across "
                 f"{len(surface.slices)} expiries"),
    )


# ---------------------------------------------------------------------------
# 2. Repair with all three models
# ---------------------------------------------------------------------------
def run_repair(surface) -> dict[str, object]:
    from arbfree_vol.repair.engine import repair

    # Pre-repair arbitrage report — used for the violations bar chart so the
    # plot always shows a real distribution (post-repair SVI is often 0).
    from arbfree_vol.arbitrage.quote_detect import detect
    before_report = detect(surface)

    reports: dict[str, object] = {}
    configs = [
        ("SVI",   {"use_ssvi": False, "use_sabr": False}),
        ("eSSVI", {"use_ssvi": True,  "use_sabr": False}),
        ("SABR",  {"use_ssvi": False, "use_sabr": True}),
    ]
    print("\n  Model      violations_before  violations_after  rejected  slices  med_RMSE")
    print("  " + "-" * 68)
    for label, kw in configs:
        rep = repair(surface, **kw)
        reports[label] = rep
        m = rep.metrics
        med_rmse = (
            float(np.median([s.rmse for s in rep.fitted_slices]))
            if rep.fitted_slices else 0.0
        )
        print(f"  {label:<8} {m.n_violations_before:>10} {m.n_violations_after:>12} "
              f"{m.n_rejected:>8} {len(rep.fitted_slices):>6} {med_rmse:.4f}")
    return reports, before_report


# ---------------------------------------------------------------------------
# 3. FittedSurface + Dupire + Greeks
# ---------------------------------------------------------------------------
def build_and_analyze(reports, surface) -> tuple[object, list[float], list[float], object]:
    """FittedSurface from the eSSVI report, then Dupire + Greeks grids."""
    from arbfree_vol.surface.interpolate import build_fitted_surface
    from arbfree_vol.pricing.local_vol import dupire

    essvi = reports["eSSVI"]
    fs = build_fitted_surface(essvi)
    fallback_Ts: list[float] = list(essvi.fallback_slices)
    if fallback_Ts:
        print(f"\n  eSSVI fallback slices (H&M arb-free fit failed -> unconstrained): "
              f"{[f'{T:.4f}' for T in fallback_Ts]}")
    else:
        print("\n  eSSVI: no fallback slices were reported "
              "(all expiries fitted on the hard-constrained path).")

    spot = surface.spot
    strikes = list(np.linspace(spot * 0.85, spot * 1.15, 20))

    # Maturities from the fitted surface (pad to >= 3 with mid-points).
    mat_base = sorted(sl.expiry_time for sl in fs.fitted_slices)
    if not mat_base:
        raise ValueError(
            "No fitted expiries survived repair — cannot build a Dupire "
            "grid.  Try more expiries (--max-expiries) or a different symbol."
        )
    maturities = list(mat_base)
    if len(mat_base) < 3:
        seen: set[float] = set()
        maturities = []
        for i in range(len(mat_base) - 1):
            T1, T2 = mat_base[i], mat_base[i + 1]
            for t in (T1, (T1 + T2) / 2.0):
                if round(t, 10) not in seen:
                    seen.add(round(t, 10))
                    maturities.append(t)
        maturities.append(mat_base[-1])

    lv = dupire(fs, strikes, maturities, fallback_slices=fallback_Ts or None)
    print(f"  Dupire grid: {len(strikes)} strikes x {len(maturities)} maturities")
    return fs, strikes, maturities, lv


# ---------------------------------------------------------------------------
# 4. Plots
# ---------------------------------------------------------------------------
def save_plots(reports, surface, fs, strikes, maturities, lv, before_report) -> None:
    from arbfree_vol.viz.surface import plot_surface, plot_iv_heatmap
    from arbfree_vol.viz.smiles import plot_smiles
    from arbfree_vol.viz.local_vol import plot_dupire_heatmap
    from arbfree_vol.viz.risk import plot_greeks_heatmap
    from arbfree_vol.viz.violations import plot_violations_bar

    fallback_Ts = list(reports["eSSVI"].fallback_slices)

    print("\nSaving plots...")
    plots = [
        ("yfinance_demo_surface.png",
         plot_surface(list(reports["SVI"].fitted_slices))),
        ("yfinance_demo_smiles.png",
         plot_smiles(surface, list(reports["eSSVI"].fitted_slices))),
        ("yfinance_demo_iv_heatmap.png",
         plot_iv_heatmap(fs, symbol=SYMBOL, fallback_slices=fallback_Ts)),
        ("yfinance_demo_dupire.png",
         plot_dupire_heatmap(lv, symbol=SYMBOL, fallback_slices=fallback_Ts)),
        ("yfinance_demo_greeks.png",
         plot_greeks_heatmap(fs, strikes, maturities, symbol=SYMBOL,
                             fallback_slices=fallback_Ts)),
        ("yfinance_demo_violations.png",
         plot_violations_bar(before_report)),
    ]
    for name, fig in plots:
        fig.savefig(str(_OUT / name), dpi=150)
        print(f"  saved: {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print()
    print(" " + "=" * 60)
    print("   arbfree-vol-surface  --  yfinance pipeline demo")
    print("   " + ("LIVE data" if not OFFLINE else "SYNTHETIC data (no network)"))
    print(" " + "=" * 60)

    # 1. Data
    bundle = _build_synthetic() if OFFLINE else _fetch_live()
    surface = bundle.surface
    print(f"\n  {bundle.summary}")
    print(f"  Spot={surface.spot:.2f}, r={surface.risk_free:.4f}, "
          f"q={surface.div_yield:.4f}")
    if bundle.quality_drops:
        reasons = Counter()
        for d in bundle.quality_drops:
            for part in d.reason.split("; "):
                reasons[part.split("=")[0]] += 1
        print(f"  Quality drops: {len(bundle.quality_drops)} "
              f"({', '.join(f'{k}={v}' for k, v in reasons.most_common())})")
    if bundle.rejected:
        rules = Counter(r.rule.value for r in bundle.rejected)
        print(f"  Cleaning rejects: {len(bundle.rejected)} "
              f"({', '.join(f'{k}={v}' for k, v in rules.most_common())})")

    # 2. Repair
    _sep("Repair: SVI / eSSVI / SABR")
    reports, before_report = run_repair(surface)

    # 3. FittedSurface -> Dupire -> Greeks
    _sep("Fitted surface, Dupire local vol, Greeks")
    fs, strikes, maturities, lv = build_and_analyze(reports, surface)

    # 4. Plots
    _sep("Plots")
    save_plots(reports, surface, fs, strikes, maturities, lv, before_report)

    print(f"\nDone. 6 plots saved to {_OUT / 'yfinance_demo_*.png'}")
    print(f"Re-run with:  python {Path(__file__).name} [--symbol QQQ] [--offline]")


if __name__ == "__main__":
    main()
