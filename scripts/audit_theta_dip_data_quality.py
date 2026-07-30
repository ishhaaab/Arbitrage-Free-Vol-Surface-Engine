"""Audit: data quality at SPY eSSVI fallback vs non-fallback expiries.

Fetches live SPY option chain data, identifies which expiries take the
eSSVI fallback path, and computes ATM-strike data quality metrics for
every expiry.  Prints a comparison table and writes findings to
docs/issues.md under Issue #15.

Runs the audit twice:
1. Filter OFF (disable_quality_filter=True) — TRUE BASELINE
2. Filter ON (default thresholds: OI>=10, spread<=50%)

Reports per-expiry OI<10 drop breakdown and fitted-slice count comparison.

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

def fetch_spy_data(disable_quality_filter: bool = False):
    """Fetch SPY option data via yfinance, same approach as the demo."""
    surface, rejected, quality_drops = fetch_chain(
        "SPY",
        max_expiries=40,
        min_T_years=7.0 / 365.0,
        disable_quality_filter=disable_quality_filter,
    )
    return surface, rejected, quality_drops


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


def compute_per_expiry_oi_drops(
    chain_calls,
    chain_puts,
    spot: float,
    min_oi: int = 10,
    atm_band: float = 0.05,
) -> dict:
    """Count OI<min_oi drops for ATM strikes in a single expiry.

    Returns dict with: total_strikes, oi_dropped, drop_rate.
    """
    import pandas as pd

    chain = pd.concat([chain_calls, chain_puts], ignore_index=True)
    lower = spot * (1.0 - atm_band)
    upper = spot * (1.0 + atm_band)
    atm = chain[(chain["strike"] >= lower) & (chain["strike"] <= upper)].copy()

    if atm.empty:
        return {"total_strikes": 0, "oi_dropped": 0, "drop_rate": 0.0}

    oi = atm["openInterest"].fillna(0)
    total = len(atm)
    dropped = int((oi < min_oi).sum())
    return {
        "total_strikes": total,
        "oi_dropped": dropped,
        "drop_rate": dropped / total if total > 0 else 0.0,
    }


# ── Single audit run ────────────────────────────────────────────────

def _run_single_audit(
    label: str,
    surface,
    quality_drops,
    spot: float,
    ticker,
) -> dict:
    """Run audit for a single fetch result and return structured results."""
    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")
    print(f"  Spot: {spot:.2f}")
    print(f"  Slices: {len(surface.slices)}")
    print(f"  Quotes: {sum(len(s.quotes) for s in surface.slices)}")
    print(f"  Quality drops: {len(quality_drops)}")

    # Identify fallback slices via sequential eSSVI fit
    slices_data = extract_slice_data(surface)
    result = fit_ssvi_surface_sequential(slices_data)

    fallback_Ts = set(result.fallback_slices)
    failed_Ts = set(result.failed_slices)
    n_fitted = len(result.fitted_slices)
    print(f"  Fitted: {n_fitted} slices")
    print(f"  Fallback: {len(result.fallback_slices)} slices — T = "
          f"{[f'{T:.4f}' for T in result.fallback_slices]}")
    print(f"  Failed: {len(result.failed_slices)} slices")

    # Per-expiry data quality metrics
    from datetime import date, timedelta

    rows = []
    for sl in sorted(surface.slices, key=lambda s: s.expiry_time):
        T = sl.expiry_time
        is_fallback = T in fallback_Ts
        is_failed = T in failed_Ts

        ref = date.today()
        exp_date = ref + timedelta(days=int(round(T * 365.0)))
        exp_str = exp_date.isoformat()

        try:
            chain = ticker.option_chain(exp_str)
            metrics = compute_atm_quality_metrics(chain.calls, chain.puts, spot)
            oi_info = compute_per_expiry_oi_drops(chain.calls, chain.puts, spot)
        except Exception:
            metrics = {
                "median_OI": float("nan"),
                "median_volume": float("nan"),
                "median_bid_ask_pct": float("nan"),
                "zero_vol_count": float("nan"),
                "zero_oi_count": float("nan"),
                "zero_quote_count": float("nan"),
                "n_atm_strikes": 0,
            }
            oi_info = {"total_strikes": 0, "oi_dropped": 0, "drop_rate": 0.0}

        tag = "FALLBACK" if is_fallback else ("FAILED" if is_failed else "OK")
        rows.append({
            "T": T,
            "tag": tag,
            **metrics,
            **oi_info,
        })

    # Print per-expiry table
    print(f"\n  {'=' * 110}")
    print(f"  {'T':>8}  {'Status':>8}  {'n_ATM':>6}  {'med_OI':>10}  "
          f"{'med_vol':>10}  {'med_sprd%':>10}  {'OI<10':>6}  "
          f"{'total':>6}  {'drop%':>6}")
    print(f"  {'-' * 98}")

    for r in rows:
        print(f"  {r['T']:>8.4f}  {r['tag']:>8}  {r['n_atm_strikes']:>6}  "
              f"{r['median_OI']:>10.0f}  {r['median_volume']:>10.0f}  "
              f"{r['median_bid_ask_pct']:>10.2f}  "
              f"{r['oi_dropped']:>6}  {r['total_strikes']:>6}  "
              f"{r['drop_rate']:>6.1%}")

    return {
        "rows": rows,
        "fallback_rows": [r for r in rows if r["tag"] == "FALLBACK"],
        "ok_rows": [r for r in rows if r["tag"] == "OK"],
        "n_fitted": n_fitted,
        "fallback_slices": result.fallback_slices,
        "failed_slices": result.failed_slices,
        "n_quality_drops": len(quality_drops),
    }


# ── Main audit ──────────────────────────────────────────────────────

def run_audit():
    """Run the full data quality audit with both filter states."""
    print("=" * 72)
    print("  Data Quality Audit: SPY eSSVI fallback vs non-fallback expiries")
    print("  (Filter OFF vs Filter ON comparison)")
    print("=" * 72)

    import yfinance as yf
    ticker = yf.Ticker("SPY")

    # Step 1: Fetch data with filter OFF (true baseline)
    print("\n[1/4] Fetching SPY data with filter DISABLED...")
    surface_raw, _, quality_drops_raw = fetch_spy_data(disable_quality_filter=True)
    spot = surface_raw.spot

    # Step 2: Fetch data with filter ON (default thresholds)
    print("[2/4] Fetching SPY data with filter ENABLED (OI>=10, spread<=50%)...")
    surface_filtered, _, quality_drops_filtered = fetch_spy_data(disable_quality_filter=False)

    # Step 3: Run audit for both
    print("[3/4] Running eSSVI fit and per-expiry analysis...")
    raw_result = _run_single_audit(
        "FILTER OFF (raw yfinance data — true baseline)",
        surface_raw, quality_drops_raw, spot, ticker,
    )
    filtered_result = _run_single_audit(
        "FILTER ON (OI>=10, spread<=50% — default thresholds)",
        surface_filtered, quality_drops_filtered, spot, ticker,
    )

    # Step 4: Comparison summary
    print(f"\n{'=' * 72}")
    print("  COMPARISON SUMMARY")
    print(f"{'=' * 72}")

    print(f"\n  Fitted slices:")
    print(f"    Filter OFF: {raw_result['n_fitted']}")
    print(f"    Filter ON:  {filtered_result['n_fitted']}")
    delta = raw_result['n_fitted'] - filtered_result['n_fitted']
    if delta > 0:
        print(f"    ⚠ {delta} slice(s) lost after filtering")
    elif delta < 0:
        print(f"    (filter gained {-delta} slice(s))")
    else:
        print(f"    (no change)")

    print(f"\n  Fallback slices:")
    print(f"    Filter OFF: {len(raw_result['fallback_slices'])} — "
          f"{[f'{T:.4f}' for T in raw_result['fallback_slices']]}")
    print(f"    Filter ON:  {len(filtered_result['fallback_slices'])} — "
          f"{[f'{T:.4f}' for T in filtered_result['fallback_slices']]}")

    print(f"\n  Quality drops:")
    print(f"    Filter OFF: {raw_result['n_quality_drops']}")
    print(f"    Filter ON:  {filtered_result['n_quality_drops']}")

    # Per-expiry OI<10 drop analysis (from filter-ON run)
    print(f"\n  Per-expiry OI<10 drop analysis (filter ON):")
    print(f"  {'T':>8}  {'Status':>8}  {'OI<10':>6}  {'total':>6}  {'drop%':>6}")
    print(f"  {'-' * 42}")
    oi_drops = []
    for r in filtered_result["rows"]:
        print(f"  {r['T']:>8.4f}  {r['tag']:>8}  {r['oi_dropped']:>6}  "
              f"{r['total_strikes']:>6}  {r['drop_rate']:>6.1%}")
        oi_drops.append(r["drop_rate"])

    if oi_drops:
        mean_drop = np.mean(oi_drops)
        std_drop = np.std(oi_drops)
        print(f"\n  Drop rate stats: mean={mean_drop:.1%}, std={std_drop:.1%}")
        if std_drop < 0.10:
            print(f"  => Drop rate is roughly UNIFORM across expiries.")
            print(f"     This is a general data-thinning effect, NOT targeted")
            print(f"     at fallback expiries specifically.")
        else:
            fb_drops = [r["drop_rate"] for r in filtered_result["fallback_rows"]]
            ok_drops = [r["drop_rate"] for r in filtered_result["ok_rows"]]
            if fb_drops and ok_drops:
                mean_fb = np.mean(fb_drops)
                mean_ok = np.mean(ok_drops)
                print(f"  => Drop rate varies: fallback mean={mean_fb:.1%}, "
                      f"OK mean={mean_ok:.1%}")

    # Interpretation
    print(f"\n{'=' * 72}")
    print("  INTERPRETATION")
    print(f"{'=' * 72}")

    if raw_result['n_fitted'] > 0 and filtered_result['n_fitted'] > 0:
        fb_before = len(raw_result['fallback_slices'])
        fb_after = len(filtered_result['fallback_slices'])
        if fb_after < fb_before:
            print(f"  Filter reduced fallback slices from {fb_before} to {fb_after}.")
            print(f"  The OI<10 filter removes thin data that causes non-monotonic theta.")
        elif fb_after == fb_before:
            print(f"  Filter did NOT change the number of fallback slices ({fb_before}).")
            print(f"  The fallback condition is driven by factors beyond OI alone.")
        else:
            print(f"  Filter INCREASED fallback slices from {fb_before} to {fb_after}.")
            print(f"  Unexpected — investigate further.")

    return {
        "raw": raw_result,
        "filtered": filtered_result,
        "spot": spot,
    }


# ── Write findings ──────────────────────────────────────────────────

def write_findings_to_issues(audit_result):
    """Update docs/issues.md Issue #15 with corrected audit findings."""
    issues_path = Path(__file__).resolve().parent.parent / "docs" / "issues.md"
    content = issues_path.read_text(encoding="utf-8")

    raw = audit_result["raw"]
    filtered = audit_result["filtered"]
    spot = audit_result["spot"]

    # Build per-expiry drop table
    per_expiry_lines = []
    for r in filtered["rows"]:
        per_expiry_lines.append(
            f"| {r['T']:.4f} | {r['tag']} | {r['oi_dropped']} | "
            f"{r['total_strikes']} | {r['drop_rate']:.1%} |"
        )

    # Drop rate stats
    oi_drops = [r["drop_rate"] for r in filtered["rows"]]
    mean_drop = float(np.mean(oi_drops)) if oi_drops else 0.0
    std_drop = float(np.std(oi_drops)) if oi_drops else 0.0
    uniform_note = ""
    if std_drop < 0.10:
        uniform_note = (
            "The OI<10 drop rate is roughly uniform across all expiries "
            f"(mean={mean_drop:.1%}, std={std_drop:.1%}), indicating a "
            "**general data-thinning effect** rather than a targeted artifact "
            "at fallback expiries."
        )

    # Build the new section
    lines = [
        "",
        "### Data quality audit — corrected (Issue #15 follow-up)",
        "",
        "The audit was re-run with an explicit filter-disable path to establish",
        "a true baseline (raw yfinance data, no quality filter applied).",
        "",
        f"**Spot:** {spot:.2f}",
        "",
        "#### Filter OFF (true baseline) vs Filter ON (OI>=10, spread<=50%)",
        "",
        "| Metric | Filter OFF | Filter ON |",
        "|--------|------------|-----------|",
        f"| Total quotes | {sum(len(s.quotes) for s in [])} | — |",
        f"| Fitted slices | {raw['n_fitted']} | {filtered['n_fitted']} |",
        f"| Fallback slices | {len(raw['fallback_slices'])} | {len(filtered['fallback_slices'])} |",
        f"| Quality drops | {raw['n_quality_drops']} | {filtered['n_quality_drops']} |",
        "",
        "#### Per-expiry OI<10 drop breakdown (filter ON)",
        "",
        "| T | Status | OI<10 dropped | Total strikes | Drop rate |",
        "|---|--------|---------------|---------------|-----------|",
    ]
    lines.extend(per_expiry_lines)
    lines.append("")

    if uniform_note:
        lines.append(f"**{uniform_note}**")
        lines.append("")

    # Fitted-slice count comparison
    lines.extend([
        "#### Fitted-slice count comparison",
        "",
        f"- Filter OFF: **{raw['n_fitted']}** fitted slices",
        f"- Filter ON: **{filtered['n_fitted']}** fitted slices",
    ])

    delta = raw['n_fitted'] - filtered['n_fitted']
    if delta > 0:
        lines.append(
            f"- **{delta} slice(s) disappeared** after filtering — "
            "an expiry that produced a valid slice on raw data can no longer "
            "produce one because too many strikes were dropped below the "
            "5-point minimum."
        )
    elif delta < 0:
        lines.append(
            f"- Filter gained **{-delta}** additional slice(s) — "
            "the cleaner data improved calibration."
        )
    else:
        lines.append("- No change in fitted slice count.")

    lines.append("")

    # Conclusion
    if filtered['n_fitted'] > 0:
        fb_before = len(raw['fallback_slices'])
        fb_after = len(filtered['fallback_slices'])
        if fb_after < fb_before:
            lines.extend([
                "**Conclusion:** The OI<10 filter reduced fallback slices from "
                f"{fb_before} to {fb_after}. The thin OI data that caused "
                "non-monotonic theta was removed by the filter.",
                "",
            ])
        elif fb_after == fb_before:
            lines.extend([
                "**Conclusion:** The OI<10 filter did NOT change the number of "
                f"fallback slices ({fb_before}). The fallback condition is driven "
                "by factors beyond open interest alone.",
                "",
            ])

    # Replace existing data quality audit section if present
    marker = "### Data quality audit"
    if marker in content:
        # Find the section and replace everything from marker to end of file
        idx = content.index(marker)
        content = content[:idx].rstrip() + "\n"

    new_content = content + "\n".join(lines) + "\n"
    issues_path.write_text(new_content, encoding="utf-8")
    print(f"\n  Wrote audit findings to {issues_path}")


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    result = run_audit()
    write_findings_to_issues(result)
