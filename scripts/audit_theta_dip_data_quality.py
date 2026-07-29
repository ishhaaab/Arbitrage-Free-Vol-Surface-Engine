"""Audit: data quality at SPY eSSVI fallback vs non-fallback expiries.

Fetches live SPY option chain data, identifies which expiries take the
eSSVI fallback path, and computes ATM-strike data quality metrics for
every expiry.  Prints a comparison table and writes findings to
docs/issues.md under Issue #15.

Usage:
    python scripts/audit_theta_dip_data_quality.py
"""

import sys
import logging
from math import log, sqrt
from pathlib import Path

import numpy as np

# Ensure project root on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arbfree_vol.ssvi.term_structure import fit_ssvi_surface_sequential
from arbfree_vol.variance import slice_total_variance
from arbfree_vol.repair.fwd_curve import estimate_forward_curve, populate_per_slice_r
from arbfree_vol.ingestion.yfinance import fetch_chain

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


# ── Data fetching ────────────────────────────────────────────────────

def fetch_spy_data():
    """Fetch SPY option data via yfinance, same approach as the demo."""
    surface, rejected, quality_drops = fetch_chain("SPY", max_expiries=40, min_T_years=7.0 / 365.0)
    return surface, rejected


# ── Slice extraction ────────────────────────────────────────────────

def extract_slice_data(surface):
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


# ── ATM data quality metrics ────────────────────────────────────────

def compute_atm_quality_metrics(
    chain_calls,
    chain_puts,
    spot: float,
    atm_band: float = 0.05,
) -> dict:
    """Compute data quality metrics for ATM strikes (within ±atm_band of spot).

    Parameters
    ----------
    chain_calls, chain_puts:
        DataFrames from yfinance ``option_chain()`` with columns:
        strike, bid, ask, volume, openInterest, impliedVolatility, etc.
    spot:
        Current underlying price.
    atm_band:
        Fraction of spot defining ATM band (e.g. 0.05 = ±5%).

    Returns
    -------
    dict with keys: median_OI, median_volume, median_bid_ask_pct,
    zero_vol_count, zero_oi_count, zero_quote_count.
    """
    import pandas as pd

    # Combine calls and puts
    chain = pd.concat([chain_calls, chain_puts], ignore_index=True)

    # Filter to ATM strikes
    lower = spot * (1.0 - atm_band)
    upper = spot * (1.0 + atm_band)
    atm = chain[(chain["strike"] >= lower) & (chain["strike"] <= upper)].copy()

    if atm.empty:
        return {
            "median_OI": 0,
            "median_volume": 0,
            "median_bid_ask_pct": float("nan"),
            "zero_vol_count": 0,
            "zero_oi_count": 0,
            "zero_quote_count": 0,
            "n_atm_strikes": 0,
        }

    # openInterest: fill NaN with 0 for counting
    oi = atm["openInterest"].fillna(0)
    vol = atm["volume"].fillna(0)

    # bid-ask spread as % of mid
    bid = atm["bid"].fillna(0)
    ask = atm["ask"].fillna(0)
    mid = (bid + ask) / 2.0
    # Avoid division by zero where mid=0
    spread_pct = np.where(mid > 0, (ask - bid) / mid * 100.0, np.nan)
    spread_pct_series = pd.Series(spread_pct, index=atm.index)

    # Count zero-quote strikes (bid==0 AND ask==0)
    zero_quote_mask = (bid == 0) & (ask == 0)

    return {
        "median_OI": float(oi.median()),
        "median_volume": float(vol.median()),
        "median_bid_ask_pct": float(spread_pct_series.median()),
        "zero_vol_count": int((vol == 0).sum()),
        "zero_oi_count": int((oi == 0).sum()),
        "zero_quote_count": int(zero_quote_mask.sum()),
        "n_atm_strikes": len(atm),
    }


# ── Main audit ──────────────────────────────────────────────────────

def run_audit():
    """Run the full data quality audit."""
    print("=" * 72)
    print("  Data Quality Audit: SPY eSSVI fallback vs non-fallback expiries")
    print("=" * 72)

    # Step 1: Fetch data
    print("\n[1/3] Fetching SPY data...")
    surface, rejected = fetch_spy_data()
    spot = surface.spot
    print(f"  Spot: {spot:.2f}")
    print(f"  Slices: {len(surface.slices)}")
    print(f"  Quotes: {sum(len(s.quotes) for s in surface.slices)}")

    # Step 2: Identify fallback slices via sequential eSSVI fit
    print("\n[2/3] Running sequential eSSVI fit to identify fallback slices...")
    slices_data = extract_slice_data(surface)
    result = fit_ssvi_surface_sequential(slices_data)

    fallback_Ts = set(result.fallback_slices)
    failed_Ts = set(result.failed_slices)
    print(f"  Fitted: {len(result.fitted_slices)} slices")
    print(f"  Fallback: {len(result.fallback_slices)} slices — T = "
          f"{[f'{T:.4f}' for T in result.fallback_slices]}")
    print(f"  Failed: {len(result.failed_slices)} slices")

    # Step 3: Compute data quality metrics per expiry
    print("\n[3/3] Computing ATM data quality metrics per expiry...")
    print(f"  ATM band: ±5% of spot ({spot:.2f})")

    import yfinance as yf
    ticker = yf.Ticker("SPY")
    ref_date = surface.slices[0].expiry_time  # just for reference

    rows = []
    for sl in sorted(surface.slices, key=lambda s: s.expiry_time):
        T = sl.expiry_time
        is_fallback = T in fallback_Ts
        is_failed = T in failed_Ts

        # Find the matching yfinance expiry string
        # We need to reconstruct it from the expiry_time
        from datetime import date, timedelta
        ref = date.today()
        exp_date = ref + timedelta(days=int(round(T * 365.0)))
        exp_str = exp_date.isoformat()

        # Try to fetch the raw chain for this expiry
        try:
            chain = ticker.option_chain(exp_str)
            metrics = compute_atm_quality_metrics(
                chain.calls, chain.puts, spot
            )
        except Exception:
            # If the expiry string doesn't match, try nearest
            metrics = {
                "median_OI": float("nan"),
                "median_volume": float("nan"),
                "median_bid_ask_pct": float("nan"),
                "zero_vol_count": float("nan"),
                "zero_oi_count": float("nan"),
                "zero_quote_count": float("nan"),
                "n_atm_strikes": 0,
            }

        tag = "FALLBACK" if is_fallback else ("FAILED" if is_failed else "OK")
        rows.append({
            "T": T,
            "tag": tag,
            **metrics,
        })

    # Print comparison table
    print(f"\n{'=' * 100}")
    print(f"  {'T':>8}  {'Status':>8}  {'n_ATM':>6}  {'med_OI':>10}  "
          f"{'med_vol':>10}  {'med_sprd%':>10}  {'0_vol':>6}  {'0_OI':>6}  {'0_qt':>6}")
    print(f"  {'-' * 88}")

    fb_rows = [r for r in rows if r["tag"] == "FALLBACK"]
    ok_rows = [r for r in rows if r["tag"] == "OK"]

    for r in rows:
        print(f"  {r['T']:>8.4f}  {r['tag']:>8}  {r['n_atm_strikes']:>6}  "
              f"{r['median_OI']:>10.0f}  {r['median_volume']:>10.0f}  "
              f"{r['median_bid_ask_pct']:>10.2f}  "
              f"{r['zero_vol_count']:>6}  {r['zero_oi_count']:>6}  "
              f"{r['zero_quote_count']:>6}")

    # Summary statistics
    def _safe_median(vals):
        clean = [v for v in vals if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))]
        return float(np.median(clean)) if clean else float("nan")

    def _safe_mean(vals):
        clean = [v for v in vals if not (isinstance(v, float) and (isinstance(v, float) and (np.isnan(v) or np.isinf(v))))]
        return float(np.mean(clean)) if clean else float("nan")

    fb_oi = [r["median_OI"] for r in fb_rows if not np.isnan(r["median_OI"])]
    ok_oi = [r["median_OI"] for r in ok_rows if not np.isnan(r["median_OI"])]
    fb_sprd = [r["median_bid_ask_pct"] for r in fb_rows if not np.isnan(r["median_bid_ask_pct"])]
    ok_sprd = [r["median_bid_ask_pct"] for r in ok_rows if not np.isnan(r["median_bid_ask_pct"])]
    fb_vol = [r["median_volume"] for r in fb_rows if not np.isnan(r["median_volume"])]
    ok_vol = [r["median_volume"] for r in ok_rows if not np.isnan(r["median_volume"])]
    fb_zv = [r["zero_vol_count"] for r in fb_rows if not np.isnan(r["zero_vol_count"])]
    ok_zv = [r["zero_vol_count"] for r in ok_rows if not np.isnan(r["zero_vol_count"])]
    fb_zq = [r["zero_quote_count"] for r in fb_rows if not np.isnan(r["zero_quote_count"])]
    ok_zq = [r["zero_quote_count"] for r in ok_rows if not np.isnan(r["zero_quote_count"])]

    med_fb_oi = _safe_median(fb_oi)
    med_ok_oi = _safe_median(ok_oi)
    med_fb_sprd = _safe_median(fb_sprd)
    med_ok_sprd = _safe_median(ok_sprd)
    med_fb_vol = _safe_median(fb_vol)
    med_ok_vol = _safe_median(ok_vol)
    sum_fb_zv = sum(fb_zv)
    sum_ok_zv = sum(ok_zv)
    sum_fb_zq = sum(fb_zq)
    sum_ok_zq = sum(ok_zq)

    print(f"\n{'=' * 100}")
    print("  SUMMARY: Fallback vs Non-Fallback")
    print(f"{'=' * 100}")
    print(f"  Fallback expiries:   {len(fb_rows)}")
    print(f"  Non-fallback (OK):   {len(ok_rows)}")
    print()
    print(f"  Median ATM open interest:")
    print(f"    Fallback:     {med_fb_oi:>10.0f}")
    print(f"    Non-fallback: {med_ok_oi:>10.0f}")
    if med_ok_oi > 0:
        print(f"    Ratio (fb/ok):{med_fb_oi / med_ok_oi:>10.2f}")
    print()
    print(f"  Median ATM volume:")
    print(f"    Fallback:     {med_fb_vol:>10.0f}")
    print(f"    Non-fallback: {med_ok_vol:>10.0f}")
    if med_ok_vol > 0:
        print(f"    Ratio (fb/ok):{med_fb_vol / med_ok_vol:>10.2f}")
    print()
    print(f"  Median ATM bid-ask spread (% of mid):")
    print(f"    Fallback:     {med_fb_sprd:>10.2f}%")
    print(f"    Non-fallback: {med_ok_sprd:>10.2f}%")
    if med_ok_sprd > 0:
        print(f"    Ratio (fb/ok):{med_fb_sprd / med_ok_sprd:>10.2f}")
    print()
    print(f"  Total zero-volume ATM strikes:")
    print(f"    Fallback:     {sum_fb_zv:>10}")
    print(f"    Non-fallback: {sum_ok_zv:>10}")
    print()
    print(f"  Total zero-quote ATM strikes (bid=ask=0):")
    print(f"    Fallback:     {sum_fb_zq:>10}")
    print(f"    Non-fallback: {sum_ok_zq:>10}")

    # Interpretation
    print(f"\n{'=' * 100}")
    print("  INTERPRETATION")
    print(f"{'=' * 100}")

    # Heuristic thresholds for "visible difference"
    oi_ratio = med_fb_oi / med_ok_oi if med_ok_oi > 0 else float("inf")
    sprd_ratio = med_fb_sprd / med_ok_sprd if med_ok_sprd > 0 else float("inf")

    data_artifact = False
    reasons = []

    if oi_ratio < 0.5:
        data_artifact = True
        reasons.append(f"OI ratio {oi_ratio:.2f} < 0.5")
    if sprd_ratio > 2.0:
        data_artifact = True
        reasons.append(f"spread ratio {sprd_ratio:.2f} > 2.0")
    if sum_fb_zv > sum_ok_zv * 1.5 and len(fb_rows) < len(ok_rows):
        data_artifact = True
        reasons.append(f"zero-volume count {sum_fb_zv} vs {sum_ok_zv}")
    if sum_fb_zq > sum_ok_zq * 1.5 and len(fb_rows) < len(ok_rows):
        data_artifact = True
        reasons.append(f"zero-quote count {sum_fb_zq} vs {sum_ok_zq}")

    if data_artifact:
        print(f"  => DATA ARTIFACT detected.")
        print(f"  Reasons: {'; '.join(reasons)}")
        print(f"  Fallback expiries show visibly thinner/wider quotes.")
        print(f"  A data-quality filter may help eliminate some fallbacks.")
        conclusion = "data_artifact"
    else:
        print(f"  => COMPARABLE DATA QUALITY between fallback and non-fallback expiries.")
        print(f"  The non-monotonic theta dip appears to be a real market feature,")
        print(f"  not a data-quality artifact.  No filter needed.")
        conclusion = "real_feature"

    return {
        "rows": rows,
        "fallback_rows": fb_rows,
        "ok_rows": ok_rows,
        "summary": {
            "med_fb_oi": med_fb_oi,
            "med_ok_oi": med_ok_oi,
            "med_fb_sprd": med_fb_sprd,
            "med_ok_sprd": med_ok_sprd,
            "oi_ratio": oi_ratio,
            "sprd_ratio": sprd_ratio,
        },
        "conclusion": conclusion,
    }


def write_findings_to_issues(audit_result):
    """Append audit findings to docs/issues.md under Issue #15."""
    issues_path = Path(__file__).resolve().parent.parent / "docs" / "issues.md"
    content = issues_path.read_text(encoding="utf-8")

    # Check if "Data quality audit" subsection already exists
    marker = "### Data quality audit"
    if marker in content:
        print(f"\n  'Data quality audit' section already exists in issues.md — "
              f"skipping write.")
        return

    # Build the findings text
    summary = audit_result["summary"]
    conclusion = audit_result["conclusion"]
    fb_rows = audit_result["fallback_rows"]
    ok_rows = audit_result["ok_rows"]

    lines = [
        "",
        "### Data quality audit (Issue #15 follow-up)",
        "",
        "An automated audit compared ATM-strike data quality metrics between",
        "fallback and non-fallback expiries on live SPY data.",
        "",
        "**Metrics (median across ATM strikes within ±5% of spot):**",
        "",
        "| Metric | Fallback expiries | Non-fallback expiries | Ratio (fb/ok) |",
        "|--------|-------------------|----------------------|---------------|",
        f"| Median open interest | {summary['med_fb_oi']:.0f} | {summary['med_ok_oi']:.0f} | {summary['oi_ratio']:.2f} |",
        f"| Median bid-ask spread (% of mid) | {summary['med_fb_sprd']:.2f}% | {summary['med_ok_sprd']:.2f}% | {summary['sprd_ratio']:.2f} |",
        f"| Number of fallback expiries | {len(fb_rows)} | — | — |",
        f"| Number of OK expiries | — | {len(ok_rows)} | — |",
        "",
    ]

    if conclusion == "data_artifact":
        lines += [
            "**Conclusion: Data quality artifact.** Fallback expiries show visibly",
            "thinner OI or wider bid-ask spreads compared to non-fallback expiries.",
            "A data-quality filter (min OI, max spread) applied before building",
            "MarketSlices could eliminate some fallback expiries.",
            "",
        ]
    else:
        lines += [
            "**Conclusion: Real market feature.** Data quality is comparable between",
            "fallback and non-fallback expiries. The non-monotonic ATM total variance",
            "(theta) dip is a genuine feature of SPY options, likely driven by event",
            "risk concentration in specific near-term expiries.  No data-quality filter",
            "is needed; the fallback behavior is the correct response to a fundamental",
            "limitation of the H&M Prop 3.1 constraint formulation.",
            "",
        ]

    # Append after the last line of Issue #15 (end of file or before next issue)
    # Find the end of Issue #15 — it's the last issue in the file
    new_content = content.rstrip() + "\n" + "\n".join(lines) + "\n"
    issues_path.write_text(new_content, encoding="utf-8")
    print(f"\n  Wrote audit findings to {issues_path}")


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_audit()
    write_findings_to_issues(result)
