"""Diagnose which H&M Prop 3.1 sub-condition fails for each eSSVI fallback slice.

Fetches live option chain data from multiple sources, runs the eSSVI
sequential fit, and for each fallback slice reports which of the three
H&M sub-conditions (theta, chi, ratio) fails. This determines whether
Issue #15 is data-driven (theta/chi violations are data artifacts) or
model-driven (ratio violations are structural H&M limitations).

Uses `verify_hm_condition_breakdown` with `fitted_slices_prev` to ensure
consecutive fallbacks are attributed to the correct predecessor (the
last hard-constrained fit, not the position-based predecessor).

This is a RESEARCH / DIAGNOSTIC tool, not part of the library test suite.
All numbers are snapshot-in-time: live fetches on the run date, fitted
with the scipy optimizer at that moment.

LIMITATION: the ratio condition is only evaluated when chi genuinely
increases.  When chi is flat or decreasing, the ratio is reported as N/A
(undefined) — the old clamped denominator manufactured a huge, misleading
ratio, and a chi dip is a PRIMARY data-driven failure, not evidence about
the model's ratio condition.  A "ratio" count below therefore means a
genuine ratio violation with monotone (increasing) chi, never a derived
consequence of a chi dip.

Usage:
    python scripts/diagnose_fallback_subcondition.py
"""

import sys
import logging
from pathlib import Path
from collections import Counter

# Ensure project root on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arbfree_vol.ssvi.term_structure import (
    fit_ssvi_surface_sequential,
    verify_hm_condition_breakdown,
)
from arbfree_vol.variance import slice_total_variance
from arbfree_vol.forward import estimate_forward_curve, populate_per_slice_r

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


# ── Data fetching (mirrors audit_theta_dip_data_quality.py) ──────────

def fetch_yf_data(symbol: str, disable_quality_filter: bool = False):
    """Fetch option data via yfinance."""
    from arbfree_vol.ingestion.yahoo import fetch_chain as yf_fetch_chain
    surface, rejected, quality_drops = yf_fetch_chain(
        symbol,
        max_expiries=40,
        min_T_years=7.0 / 365.0,
        disable_quality_filter=disable_quality_filter,
    )
    return surface, rejected, quality_drops


def fetch_openbb_data(symbol: str, disable_quality_filter: bool = False):
    """Fetch option data via OpenBB (yfinance provider)."""
    try:
        from arbfree_vol.ingestion.openbb import fetch_chain as obb_fetch_chain
    except ImportError:
        return None, None, None
    surface, rejected, quality_drops = obb_fetch_chain(
        symbol,
        max_expiries=40,
        min_T_years=7.0 / 365.0,
        disable_quality_filter=disable_quality_filter,
        provider="yfinance",
    )
    return surface, rejected, quality_drops


# ── Slice extraction (mirrors audit_theta_dip_data_quality.py) ───────

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
        from math import log
        pts = [(log(strike / F), w) for strike, w in strike_w.items()]
        pts.sort()
        slices_data.append((sl.expiry_time, pts))
    return slices_data


# ── Per-source diagnostic ────────────────────────────────────────────

def _build_slice_row(fb_T: float, entry: dict) -> dict:
    """Build one fallback-slice result row and print its diagnostic line.

    The ratio is only defined when chi genuinely increases.  When chi is
    flat or decreasing the ratio is N/A (undefined) — a chi dip is the
    PRIMARY failure, and any ratio number would be a derived, misleading
    diagnostic.
    """
    prev_T = entry["prev_T"]
    theta_self = entry["theta_self"]
    theta_prev = entry["theta_prev"]
    chi_self = entry["chi_self"]
    chi_prev = entry["chi_prev"]
    ratio_val = entry["ratio_value"]
    failing = entry["failing_conditions"]

    # Compute percentage drops for display
    theta_drop = (theta_prev - theta_self) / theta_prev * 100 if theta_prev > 0 else 0
    chi_drop = (chi_prev - chi_self) / chi_prev * 100 if chi_prev > 0 else 0

    ratio_defined = ratio_val is not None
    ratio_str = f"{ratio_val:>8.4f}" if ratio_defined else "     N/A"

    # Format fail description — primary failures (theta/chi) first,
    # then the derived ratio diagnostic (only present when chi
    # increases and the slope condition genuinely fails).
    fail_parts = []
    for cond in failing:
        if cond == "theta":
            fail_parts.append(f"theta (drops {theta_drop:.0f}%)")
        elif cond == "chi":
            fail_parts.append(f"chi (drops {chi_drop:.0f}%)")
        elif cond == "ratio":
            fail_parts.append(f"ratio ({ratio_val:.4f} > 1)")
    if not ratio_defined and not entry["chi_ok"]:
        fail_parts.append("ratio N/A (chi non-monotonic)")
    fail_str = ", ".join(fail_parts) if fail_parts else "(none — should not be fallback?)"

    print(f"  {fb_T:>8.4f}  {prev_T:>8.4f}  {theta_self:>10.6f}  "
          f"{chi_self:>10.6f}  {ratio_str}  FAILS: {fail_str}")

    return {
        "T": fb_T,
        "prev_T": prev_T,
        "theta_self": theta_self,
        "theta_prev": theta_prev,
        "theta_ok": entry["theta_ok"],
        "chi_self": chi_self,
        "chi_prev": chi_prev,
        "chi_ok": entry["chi_ok"],
        "ratio_value": ratio_val,
        "ratio_ok": entry["ratio_ok"],
        "failing_conditions": failing,
    }


def diagnose_source(label: str, surface) -> dict | None:
    """Run eSSVI fit and H&M sub-condition breakdown for one data source.

    Returns a dict with per-slice breakdown and aggregate counts,
    or None if the surface is empty.
    """
    if surface is None:
        return None

    print(f"\n{'=' * 72}")
    print(f"  {label}")
    print(f"{'=' * 72}")
    print(f"  Spot: {surface.spot:.2f}")
    print(f"  Slices: {len(surface.slices)}")
    print(f"  Quotes: {sum(len(s.quotes) for s in surface.slices)}")

    # Extract slice data and run sequential fit
    slices_data = extract_slice_data(surface)
    result = fit_ssvi_surface_sequential(slices_data)

    n_fallback = len(result.fallback_slices)
    n_fitted = len(result.fitted_slices)
    print(f"  Fitted: {n_fitted} slices")
    print(f"  Fallback: {n_fallback} slices")

    if n_fallback == 0:
        print("  No fallback slices — nothing to diagnose.")
        return {
            "label": label,
            "n_fitted": n_fitted,
            "n_fallback": 0,
            "per_slice": [],
            "aggregate": {"theta": 0, "chi": 0, "ratio": 0, "multi": 0},
        }

    # Run the breakdown with fitted_slices_prev (correct predecessor tracking)
    breakdown = verify_hm_condition_breakdown(
        result.fitted_slices,
        result.fitted_slices_prev,
    )

    # Index breakdown by slice_T for lookup
    breakdown_by_T = {entry["slice_T"]: entry for entry in breakdown}

    # Per-slice output for fallback slices
    per_slice_results = []
    counter = Counter()  # counts by sub-condition
    multi_count = 0

    print(f"\n  Per-slice breakdown:")
    print(f"  {'T':>8}  {'prev_T':>8}  {'theta':>10}  {'chi':>10}  "
          f"{'ratio':>8}  {'FAILS':<30}")
    print(f"  {'-' * 78}")

    for fb_T in result.fallback_slices:
        entry = breakdown_by_T.get(fb_T)
        if entry is None:
            # No entry found — shouldn't happen for a fallback slice
            # (it has a predecessor), but handle gracefully
            print(f"  {fb_T:>8.4f}  {'N/A':>8}  {'N/A':>10}  {'N/A':>10}  "
                  f"{'N/A':>8}  (no breakdown entry)")
            continue

        row = _build_slice_row(fb_T, entry)
        per_slice_results.append(row)

        # Count each failing condition
        for cond in entry["failing_conditions"]:
            counter[cond] += 1
        if len(entry["failing_conditions"]) > 1:
            multi_count += 1

    # Aggregate breakdown
    aggregate = {
        "theta": counter.get("theta", 0),
        "chi": counter.get("chi", 0),
        "ratio": counter.get("ratio", 0),
        "multi": multi_count,
    }

    print(f"\n  Breakdown:")
    print(f"    theta violations: {aggregate['theta']}/{n_fallback}")
    print(f"    chi violations:   {aggregate['chi']}/{n_fallback}")
    print(f"    ratio violations: {aggregate['ratio']}/{n_fallback}")
    print(f"    multi-condition:  {aggregate['multi']}/{n_fallback}")
    print(f"    (ratio is only counted where chi increases; where chi dips")
    print(f"     the ratio is N/A — a derived consequence, not a separate")
    print(f"     model failure)")

    return {
        "label": label,
        "n_fitted": n_fitted,
        "n_fallback": n_fallback,
        "per_slice": per_slice_results,
        "aggregate": aggregate,
    }


# ── Cross-source comparison ──────────────────────────────────────────

def print_comparison(all_results: dict[str, dict | None]):
    """Print a cross-source comparison table."""
    print(f"\n{'=' * 72}")
    print("  CROSS-SOURCE COMPARISON: H&M sub-condition breakdown")
    print(f"{'=' * 72}")
    print(f"\n  {'Source':<30} {'FB':>4} {'theta':>6} {'chi':>6} "
          f"{'ratio':>6} {'multi':>6}")
    print(f"  {'-' * 60}")

    for label, res in all_results.items():
        if res is None:
            print(f"  {label:<30} {'N/A':>4} {'N/A':>6} {'N/A':>6} "
                  f"{'N/A':>6} {'N/A':>6}")
            continue
        agg = res["aggregate"]
        fb = res["n_fallback"]
        print(f"  {label:<30} {fb:>4} {agg['theta']:>6} {agg['chi']:>6} "
              f"{agg['ratio']:>6} {agg['multi']:>6}")

    # Interpretation
    print(f"\n  Interpretation:")
    print(f"  - theta violations (PRIMARY) = data wants lower ATM variance")
    print(f"    at this maturity than the predecessor. Likely a data artifact")
    print(f"    (event risk, microstructure noise) or a genuine short-end feature.")
    print(f"  - chi violations (PRIMARY) = theta*psi dips, often correlated")
    print(f"    with theta dips (if theta drops, chi likely drops too).")
    print(f"  - ratio violations (DERIVED) = theta and chi are both increasing,")
    print(f"    but the cross-slice slope condition |(rho*chi)'| / chi' <= 1")
    print(f"    fails. This is a structural H&M limitation.")
    print(f"  - ratio is only evaluated where chi increases.  Where chi dips,")
    print(f"    the ratio is N/A (undefined) — a chi dip is the primary failure,")
    print(f"    and the old clamped denominator produced a huge misleading ratio.")

    # Key finding
    has_data = {k: v for k, v in all_results.items() if v is not None}
    if has_data:
        total_theta = sum(v["aggregate"]["theta"] for v in has_data.values())
        total_chi = sum(v["aggregate"]["chi"] for v in has_data.values())
        total_ratio = sum(v["aggregate"]["ratio"] for v in has_data.values())
        total_fb = sum(v["n_fallback"] for v in has_data.values())

        print(f"\n  Overall: {total_fb} total fallbacks across {len(has_data)} sources")
        print(f"    theta: {total_theta} ({total_theta/max(total_fb,1)*100:.0f}%)")
        print(f"    chi:   {total_chi} ({total_chi/max(total_fb,1)*100:.0f}%)")
        print(f"    ratio: {total_ratio} ({total_ratio/max(total_fb,1)*100:.0f}%)  "
              f"(only where chi increases; N/A where chi dips)")

        if total_ratio == 0 and total_theta > 0:
            print(f"\n  => ALL fallbacks are data-driven (theta/chi violations).")
            print(f"     No structural H&M limitations found — the ratio condition")
            print(f"     is satisfied wherever chi increases.")
        elif total_ratio > 0 and total_theta == 0:
            print(f"\n  => ALL fallbacks are model-driven (ratio violations).")
            print(f"     The H&M slope condition is the binding constraint.")
        else:
            print(f"\n  => Mixed: some data-driven (theta/chi) and some")
            print(f"     model-driven (ratio) fallbacks.")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  H&M Sub-Condition Diagnostic: which condition drives each fallback?")
    print("=" * 72)

    all_results: dict[str, dict | None] = {}

    # ── Source 1: yfinance + SPY (raw) ───────────────────────────────
    print("\n[1/3] yfinance + SPY (raw, filter OFF)")
    print("-" * 40)
    try:
        spy_surface, _, _ = fetch_yf_data("SPY", disable_quality_filter=True)
        all_results["yfinance/SPY (raw)"] = diagnose_source(
            "yfinance/SPY (raw)", spy_surface
        )
    except Exception as exc:
        print(f"  SPY fetch failed: {exc}")
        all_results["yfinance/SPY (raw)"] = None

    # ── Source 2: yfinance + SPX (raw) ───────────────────────────────
    print("\n[2/3] yfinance + ^SPX (raw, filter OFF)")
    print("-" * 40)
    try:
        spx_surface, _, _ = fetch_yf_data("^SPX", disable_quality_filter=True)
        all_results["yfinance/^SPX (raw)"] = diagnose_source(
            "yfinance/^SPX (raw)", spx_surface
        )
    except Exception as exc:
        print(f"  SPX fetch failed: {exc}")
        all_results["yfinance/^SPX (raw)"] = None

    # ── Source 3: OpenBB + SPY (raw) ─────────────────────────────────
    print("\n[3/3] OpenBB + SPY (raw, filter OFF)")
    print("-" * 40)
    try:
        obb_surface, _, _ = fetch_openbb_data("SPY", disable_quality_filter=True)
        if obb_surface is not None:
            all_results["OpenBB/SPY (raw)"] = diagnose_source(
                "OpenBB/SPY (raw)", obb_surface
            )
        else:
            print("  OpenBB not available — skipping.")
            all_results["OpenBB/SPY (raw)"] = None
    except Exception as exc:
        print(f"  OpenBB fetch failed: {exc}")
        all_results["OpenBB/SPY (raw)"] = None

    # ── Cross-source comparison ──────────────────────────────────────
    print_comparison(all_results)

    return all_results


if __name__ == "__main__":
    results = main()
