"""Audit: data quality at eSSVI fallback vs non-fallback expiries.

Fetches live option chain data, identifies which expiries take the eSSVI
fallback path, and computes ATM-strike data quality metrics for every
expiry.  Prints comparison tables and writes findings to docs/issues.md
under Issue #15.

This is a RESEARCH / DIAGNOSTIC tool, not part of the library test suite.
All numbers are snapshot-in-time: they come from live fetches on the day
the script is run and vary day to day.  The raw vs filtered runs are
SEPARATE live fetches (differences may include market movement between
the two calls), and per-expiry option-chain metrics can be unavailable
for a given source / expiry — those are reported as N/A, never as
observed zeros.

The comparisons are UNDERLYING / INGESTION-PATH comparisons, not
provider comparisons:
1. yfinance + SPY (ETF) vs yfinance + ^SPX (index) changes the UNDERLYING
   — and with it the option market being measured — so any difference is
   an underlying effect, not a data-source effect.
2. OpenBB + SPY is configured with provider="yfinance", so OpenBB/SPY vs
   yfinance/SPY only tests the ingestion path (normalisation), NOT
   provider independence: the two paths share the same upstream provider.

For index symbols (^SPX), the per-expiry dividend yield is estimated from
put-call parity on the option chain itself (median across ATM strikes),
with a representative-ETF trailing yield (SPY for ^SPX) as fallback when
parity estimation fails.  q is NOT assumed to be zero.

For each source, runs with filter OFF (true baseline) and filter ON
(default thresholds: OI>=10, spread<=50%).  In --use-fixture mode the
filtered comparison is N/A: the fixture is a post-ingestion surface with
no raw chains to filter.

Volume is recorded as diagnostic context only — it is not a filter criterion.

Usage:
    python scripts/audit_theta_dip_data_quality.py
    python scripts/audit_theta_dip_data_quality.py --use-fixture
"""

import argparse
import sys
import logging
from math import log
from pathlib import Path

import numpy as np

# Ensure project root on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arbfree_vol.ssvi.term_structure import fit_ssvi_surface_sequential
from arbfree_vol.variance import slice_total_variance
from arbfree_vol.forward import estimate_forward_curve, populate_per_slice_r
from arbfree_vol.ingestion.yfinance import fetch_chain as yf_fetch_chain
from arbfree_vol.models.surface import VolSurface

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit eSSVI fallback counts across data sources"
    )
    parser.add_argument(
        "--use-fixture",
        action="store_true",
        help="Load SPX data from tests/fixtures/spx_sample.json instead of yfinance",
    )
    return parser.parse_args()


def load_spx_fixture() -> VolSurface:
    """Load SPX raw data from the saved fixture file."""
    import json

    from arbfree_vol.models.option import OptionType
    from arbfree_vol.models.surface import ExpirySlice, Quote

    fixture_path = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "spx_sample.json"
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    slices = []
    for sl_data in data["slices"]:
        quotes = [
            Quote(
                strike=q["strike"],
                option_type=OptionType(q["option_type"]),
                price=q["price"],
                bid=q["bid"],
                ask=q["ask"],
            )
            for q in sl_data["quotes"]
            if q["price"] is not None
        ]
        if not quotes:
            continue
        slices.append(ExpirySlice(
            expiry_time=sl_data["expiry_time"],
            risk_free=sl_data.get("risk_free"),
            div_yield=sl_data.get("div_yield"),
            quotes=quotes,
        ))
    return VolSurface(
        spot=data["spot"],
        risk_free=data["risk_free"],
        div_yield=data["div_yield"],
        slices=slices,
    )


# ── Data fetching ────────────────────────────────────────────────────

def fetch_yf_data(symbol: str, disable_quality_filter: bool = False):
    """Fetch option data via yfinance."""
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


# ── Theta monotonicity analysis ──────────────────────────────────────

def compute_theta_dip_severity(slices_data):
    """Compute theta (ATM total variance) non-monotonicity severity.

    Returns a dict with:
      - n_dips: number of consecutive-pair dips
      - max_dip_pct: maximum dip as a percentage of predecessor theta
      - mean_dip_pct: mean dip percentage (of dips only)
    """
    if len(slices_data) < 2:
        return {"n_dips": 0, "max_dip_pct": 0.0, "mean_dip_pct": 0.0}

    # Extract ATM theta (w at k=0) for each slice
    thetas = []
    for T, pts in slices_data:
        # pts is [(k, w)] sorted by k; find the point closest to k=0
        atm_w = min(pts, key=lambda p: abs(p[0]))[1]
        thetas.append((T, atm_w))

    dips = []
    for i in range(1, len(thetas)):
        prev_T, prev_w = thetas[i - 1]
        curr_T, curr_w = thetas[i]
        if curr_w < prev_w:
            dip_pct = (prev_w - curr_w) / prev_w * 100.0
            dips.append(dip_pct)

    return {
        "n_dips": len(dips),
        "max_dip_pct": max(dips) if dips else 0.0,
        "mean_dip_pct": float(np.mean(dips)) if dips else 0.0,
    }


# ── Tenor bucket breakdown ───────────────────────────────────────────

_TENOR_BUCKETS = [
    (0.10, "< 0.10y"),
    (0.25, "0.10-0.25y"),
    (0.50, "0.25-0.50y"),
    (1.00, "0.50-1.00y"),
    (2.00, "1.00-2.00y"),
    (float("inf"), "> 2.00y"),
]


def _bucket_T(T: float) -> str:
    """Map a maturity in years to a tenor bucket label."""
    for upper, label in _TENOR_BUCKETS:
        if T < upper:
            return label
    return _TENOR_BUCKETS[-1][1]


def compute_tenor_bucket_breakdown(
    surface,
    fallback_Ts: list[float],
) -> dict[str, dict[str, int]]:
    """Count fallback vs total slices per tenor bucket.

    Returns dict mapping bucket label to {"fallback": int, "total": int}.
    """
    result = {label: {"fallback": 0, "total": 0} for _, label in _TENOR_BUCKETS}
    fallback_set = set(fallback_Ts)
    for sl in surface.slices:
        bucket = _bucket_T(sl.expiry_time)
        result[bucket]["total"] += 1
        if sl.expiry_time in fallback_set:
            result[bucket]["fallback"] += 1
    return result


# ── Single audit run ────────────────────────────────────────────────

def _run_single_audit(
    label: str,
    surface,
    quality_drops,
    spot: float,
    ticker=None,
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

    # Theta dip severity
    theta_info = compute_theta_dip_severity(slices_data)
    print(f"  Theta dips: {theta_info['n_dips']}, "
          f"max={theta_info['max_dip_pct']:.1f}%, "
          f"mean={theta_info['mean_dip_pct']:.1f}%")

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

        metrics = None
        oi_info = None
        metrics_error = None
        if ticker is not None:
            try:
                chain = ticker.option_chain(exp_str)
                metrics = compute_atm_quality_metrics(chain.calls, chain.puts, spot)
                oi_info = compute_per_expiry_oi_drops(chain.calls, chain.puts, spot)
            except Exception as exc:
                metrics_error = str(exc)
                _logger.warning(
                    "option_chain fetch failed for source=%r expiry=%s T=%.4f: %s",
                    label, exp_str, T, exc,
                )
        else:
            metrics_error = "no ticker (per-expiry option chains unavailable in fixture mode)"
        metrics_available = metrics is not None
        if metrics is None:
            # Unavailable metrics are represented as None / N/A — never as
            # observed zeros (a missing chain is not evidence of zero dips).
            metrics = {
                "median_OI": None,
                "median_volume": None,
                "median_bid_ask_pct": None,
                "zero_vol_count": None,
                "zero_oi_count": None,
                "zero_quote_count": None,
                "n_atm_strikes": None,
            }
            oi_info = {
                "total_strikes": None,
                "oi_dropped": None,
                "drop_rate": None,
            }

        tag = "FALLBACK" if is_fallback else ("FAILED" if is_failed else "OK")
        rows.append({
            "T": T,
            "tag": tag,
            "metrics_available": metrics_available,
            "error": metrics_error,
            **metrics,
            **oi_info,
        })

    # Print per-expiry table
    print(f"\n  {'=' * 110}")
    print(f"  {'T':>8}  {'Status':>8}  {'n_ATM':>6}  {'med_OI':>10}  "
          f"{'med_vol':>10}  {'med_sprd%':>10}  {'OI<10':>6}  "
          f"{'total':>6}  {'drop%':>6}")
    print(f"  {'-' * 98}")

    n_unavailable = 0
    for r in rows:
        if not r["metrics_available"]:
            n_unavailable += 1
            print(f"  {r['T']:>8.4f}  {r['tag']:>8}  "
                  f"{'N/A':>6}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  "
                  f"{'N/A':>6}  {'N/A':>6}  {'N/A':>6}  "
                  f"(metrics N/A: {r['error']})")
            continue
        print(f"  {r['T']:>8.4f}  {r['tag']:>8}  {r['n_atm_strikes']:>6}  "
              f"{r['median_OI']:>10.0f}  {r['median_volume']:>10.0f}  "
              f"{r['median_bid_ask_pct']:>10.2f}  "
              f"{r['oi_dropped']:>6}  {r['total_strikes']:>6}  "
              f"{r['drop_rate']:>6.1%}")
    if n_unavailable:
        print(f"  (per-expiry option-chain metrics unavailable for "
              f"{n_unavailable}/{len(rows)} expiries; those rows are excluded "
              f"from the metric summary above)")

    # Tenor bucket breakdown
    tenor_breakdown = compute_tenor_bucket_breakdown(surface, result.fallback_slices)
    print(f"\n  {'Tenor Bucket Breakdown':}")
    print(f"  {'Bucket':<12} {'Fallback':>8} {'Total':>8}")
    print(f"  {'-' * 30}")
    for _, label in _TENOR_BUCKETS:
        tb = tenor_breakdown[label]
        if tb["total"] > 0:
            print(f"  {label:<12} {tb['fallback']:>8} {tb['total']:>8}")

    return {
        "rows": rows,
        "fallback_rows": [r for r in rows if r["tag"] == "FALLBACK"],
        "ok_rows": [r for r in rows if r["tag"] == "OK"],
        "n_fitted": n_fitted,
        "fallback_slices": result.fallback_slices,
        "failed_slices": result.failed_slices,
        "n_quality_drops": len(quality_drops),
        "theta_dips": theta_info["n_dips"],
        "theta_max_dip_pct": theta_info["max_dip_pct"],
        "tenor_breakdown": tenor_breakdown,
    }


# ── Main audit ──────────────────────────────────────────────────────

def run_audit(args=None):
    """Run the full data quality audit across multiple sources/symbols."""
    if args is None:
        args = parse_args()
    print("=" * 72)
    print("  Data Quality Audit: eSSVI fallback across underlyings / ingestion paths")
    print("=" * 72)
    print("  These are UNDERLYING / INGESTION-PATH comparisons, not provider")
    print("  comparisons: SPY vs ^SPX changes the underlying, and OpenBB is")
    print("  configured with provider='yfinance', so OpenBB/SPY vs yfinance/SPY")
    print("  does NOT test provider independence.")
    print("=" * 72)

    import yfinance as yf

    results = {}

    # ── Source 1: yfinance + SPY ─────────────────────────────────────
    print("\n[1] yfinance + SPY")
    print("-" * 40)
    ticker_spy = yf.Ticker("SPY")

    print("  Fetching with filter DISABLED...")
    spy_raw_surface, _, spy_raw_drops = fetch_yf_data("SPY", disable_quality_filter=True)
    spy_spot = spy_raw_surface.spot

    print("  Fetching with filter ENABLED (OI>=10, spread<=50%)...")
    spy_filt_surface, _, spy_filt_drops = fetch_yf_data("SPY", disable_quality_filter=False)

    # The raw and filtered runs are SEPARATE live fetches — they do not
    # share input data, so differences may include market movement between
    # the two calls, not just the filter.
    print("\n  NOTE: FILTER OFF and FILTER ON are SEPARATE live fetches;")
    print("  differences may include market movement between the two calls.")

    print("  Running eSSVI fit...")
    spy_raw_result = _run_single_audit(
        "yfinance/SPY — FILTER OFF",
        spy_raw_surface, spy_raw_drops, spy_spot, ticker_spy,
    )
    spy_filt_result = _run_single_audit(
        "yfinance/SPY — FILTER ON",
        spy_filt_surface, spy_filt_drops, spy_spot, ticker_spy,
    )
    results["yfinance_SPY"] = {
        "raw": spy_raw_result,
        "filtered": spy_filt_result,
        "spot": spy_spot,
    }

    # ── Source 2: yfinance + SPX ─────────────────────────────────────
    print("\n[2] yfinance + ^SPX (from fixture)" if args.use_fixture else "\n[2] yfinance + ^SPX")
    print("-" * 40)
    try:
        if args.use_fixture:
            print("  Using saved fixture — a post-ingestion surface, not a raw chain.")
            spx_raw_surface = load_spx_fixture()
            spx_spot = spx_raw_surface.spot
            spx_raw_drops = []
            spx_filt_surface = None
            spx_filt_drops = []
            ticker_spx = None
        else:
            ticker_spx = yf.Ticker("^SPX")
            spx_raw_surface, _, spx_raw_drops = fetch_yf_data("^SPX", disable_quality_filter=True)
            spx_spot = spx_raw_surface.spot
            spx_filt_surface, _, spx_filt_drops = fetch_yf_data("^SPX", disable_quality_filter=False)
            print("\n  NOTE: FILTER OFF and FILTER ON are SEPARATE live fetches;")
            print("  differences may include market movement between the two calls.")

        spx_raw_result = _run_single_audit(
            "yfinance/^SPX — FILTER OFF" + (" (FIXTURE)" if args.use_fixture else ""),
            spx_raw_surface, spx_raw_drops, spx_spot, ticker_spx,
        )
        if args.use_fixture:
            # No fake filtered comparison: the fixture is a post-ingestion
            # surface and the filtering comparison requires raw option chains
            # (OI / spread columns) that the fixture does not contain.
            spx_filt_result = None
            print("\n  yfinance/^SPX — FILTER ON: N/A (fixture mode)")
            print("  The fixture is a post-ingestion surface; the filtering")
            print("  comparison requires raw option chains. Run without")
            print("  --use-fixture for the filter comparison.")
        else:
            spx_filt_result = _run_single_audit(
                "yfinance/^SPX — FILTER ON",
                spx_filt_surface, spx_filt_drops, spx_spot, ticker_spx,
            )
        results["yfinance_SPX"] = {
            "raw": spx_raw_result,
            "filtered": spx_filt_result,
            "spot": spx_spot,
        }
    except Exception as exc:
        print(f"  SPX audit failed: {exc}")
        results["yfinance_SPX"] = None

    # ── Source 3: OpenBB + SPY ───────────────────────────────────────
    print("\n[3] OpenBB + SPY (provider='yfinance' — ingestion-path comparison only)")
    print("-" * 40)
    try:
        obb_raw_surface, _, obb_raw_drops = fetch_openbb_data("SPY", disable_quality_filter=True)
        if obb_raw_surface is not None:
            obb_spot = obb_raw_surface.spot
            obb_filt_surface, _, obb_filt_drops = fetch_openbb_data("SPY", disable_quality_filter=False)
            print("\n  NOTE: FILTER OFF and FILTER ON are SEPARATE live fetches;")
            print("  differences may include market movement between the two calls.")

            obb_raw_result = _run_single_audit(
                "OpenBB/SPY — FILTER OFF",
                obb_raw_surface, obb_raw_drops, obb_spot,
            )
            obb_filt_result = _run_single_audit(
                "OpenBB/SPY — FILTER ON",
                obb_filt_surface, obb_filt_drops, obb_spot,
            )
            results["openbb_SPY"] = {
                "raw": obb_raw_result,
                "filtered": obb_filt_result,
                "spot": obb_spot,
            }
        else:
            print("  OpenBB not available — skipping.")
            results["openbb_SPY"] = None
    except Exception as exc:
        print(f"  OpenBB audit failed: {exc}")
        results["openbb_SPY"] = None

    # ── Comparison table ─────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  UNDERLYING / PATH COMPARISON (not a provider comparison)")
    print(f"{'=' * 72}")
    print("  SPY vs ^SPX changes the underlying; OpenBB/SPY uses")
    print("  provider='yfinance', so it does NOT test provider independence.")
    print()

    header = (f"  {'Source':<25} {'Fitted':>7} {'Fallback':>9} "
              f"{'Drops':>6} {'ThDips':>7} {'MaxDip%':>8}")
    print(header)
    print(f"  {'-' * 65}")

    for key, label in [
        ("yfinance_SPY", "yfinance/SPY raw"),
        ("yfinance_SPY", "yfinance/SPY filt"),
        ("yfinance_SPX", "yfinance/SPX raw"),
        ("yfinance_SPX", "yfinance/SPX filt"),
        ("openbb_SPY",    "OpenBB/SPY raw"),
        ("openbb_SPY",    "OpenBB/SPY filt"),
    ]:
        is_raw = "raw" in label
        suffix = "raw" if is_raw else "filt"
        row_label = f"{key.replace('_', '/')} {suffix}"
        entry = results.get(key)
        if entry is None:
            # Source never measured — an explicit N/A row, never a zero.
            print(f"  {row_label:<25}  {'N/A':>7}  {'N/A':>9}  {'N/A':>6}  "
                  f"{'N/A':>7}  {'N/A':>8}   (source unavailable)")
            continue
        r = entry["raw"] if is_raw else entry["filtered"]
        if r is None:
            print(f"  {row_label:<25}  {'N/A':>7}  {'N/A':>9}  {'N/A':>6}  "
                  f"{'N/A':>7}  {'N/A':>8}   (run not performed: "
                  f"fixture mode has no filtered comparison)")
            continue
        print(f"  {row_label:<25} {r['n_fitted']:>7} "
              f"{len(r['fallback_slices']):>9} "
              f"{r['n_quality_drops']:>6} "
              f"{r['theta_dips']:>7} "
              f"{r['theta_max_dip_pct']:>8.1f}%")

    # ── Tenor bucket breakdown ───────────────────────────────────────
    print(f"\n{'=' * 72}")
    print("  TENOR BUCKET BREAKDOWN (fallback / total per bucket)")
    print(f"{'=' * 72}")

    for key, label in [
        ("yfinance_SPY", "yfinance/SPY raw"),
        ("yfinance_SPY", "yfinance/SPY filt"),
        ("yfinance_SPX", "yfinance/SPX raw"),
        ("yfinance_SPX", "yfinance/SPX filt"),
        ("openbb_SPY",    "OpenBB/SPY raw"),
        ("openbb_SPY",    "OpenBB/SPY filt"),
    ]:
        entry = results.get(key)
        if entry is None:
            continue
        is_raw = "raw" in label
        r = entry["raw"] if is_raw else entry["filtered"]
        if r is None:
            continue
        suffix = "raw" if is_raw else "filt"
        row_label = f"{key.replace('_', '/')} {suffix}"

        tenor = r.get("tenor_breakdown", {})
        if not tenor:
            continue

        print(f"\n  {row_label}:")
        print(f"    {'Bucket':<12} {'Fallback':>8} {'Total':>8}")
        print(f"    {'-' * 30}")
        for _, bucket_label in _TENOR_BUCKETS:
            tb = tenor.get(bucket_label, {"fallback": 0, "total": 0})
            if tb["total"] > 0:
                print(f"    {bucket_label:<12} {tb['fallback']:>8} {tb['total']:>8}")

    return results


# ── Write findings ──────────────────────────────────────────────────

def write_findings_to_issues(results):
    """Update docs/issues.md Issue #15 with the underlying/path comparison."""
    issues_path = Path(__file__).resolve().parent.parent / "docs" / "issues.md"
    content = issues_path.read_text(encoding="utf-8")

    # Build the comparison table
    lines = [
        "",
        "### Underlying / path comparison (Issue #15 follow-up)",
        "",
        "The audit compares different UNDERLYINGS and INGESTION PATHS, not",
        "independent data providers:",
        "- yfinance/SPY vs yfinance/^SPX changes the UNDERLYING (ETF vs index),",
        "  so any difference is an underlying effect, not a data-source effect;",
        "- OpenBB/SPY is configured with provider='yfinance', so OpenBB/SPY vs",
        "  yfinance/SPY only tests the ingestion path (normalisation) — it does",
        "  NOT test provider independence.",
        "",
        "All numbers are snapshot-in-time from a single calendar date and vary",
        "day-to-day.",
        "",
        "#### Underlyings / paths compared",
        "",
        "| Path | Fitted | Fallback | Quality drops | Theta dips | Max dip % |",
        "|------|--------|----------|---------------|------------|-----------|",
    ]

    for key, label in [
        ("yfinance_SPY", "yfinance/SPY"),
        ("yfinance_SPX", "yfinance/SPX"),
        ("openbb_SPY",    "OpenBB/SPY"),
    ]:
        entry = results.get(key)
        if entry is None:
            lines.append(f"| {label} | N/A (source unavailable) | N/A | N/A | N/A | N/A |")
            continue
        raw = entry["raw"]
        filt = entry["filtered"]
        lines.append(
            f"| {label} (raw) | {raw['n_fitted']} | "
            f"{len(raw['fallback_slices'])} | {raw['n_quality_drops']} | "
            f"{raw['theta_dips']} | {raw['theta_max_dip_pct']:.1f}% |"
        )
        if filt is None:
            # Mirror the console's explanation (fixture mode has no raw
            # chains, so the filtered comparison cannot be performed) in
            # the generated Issue #15 table — a bare "N/A" would read as
            # "not applicable" without saying why.
            lines.append(
                f"| {label} (filtered) | N/A (fixture has no raw chains) | "
                f"N/A | N/A | N/A | N/A |"
            )
        else:
            lines.append(
                f"| {label} (filtered) | {filt['n_fitted']} | "
                f"{len(filt['fallback_slices'])} | {filt['n_quality_drops']} | "
                f"{filt['theta_dips']} | {filt['theta_max_dip_pct']:.1f}% |"
            )

    lines.extend([
        "",
        "**Key question:** Does switching underlying (SPY ETF to ^SPX index) or",
        "ingestion path (yfinance to OpenBB) reduce theta non-monotonicity",
        "independent of the quality filter?  OpenBB uses provider='yfinance', so",
        "the OpenBB leg does not test provider independence.",
        "",
    ])

    # Determine answer from results.  A source that was not measured on
    # this run (fetch failure, fixture mode without a filtered comparison)
    # is N/A — it must NOT contribute a zero that can be compared as if it
    # were observed.  A comparative conclusion is only drawn when BOTH
    # operands were actually measured.
    def _raw_dips(key: str):
        entry = results.get(key)
        if entry is None:
            return None
        raw = entry.get("raw")
        if raw is None:
            return None
        return raw.get("theta_dips")

    spy_raw_dips = _raw_dips("yfinance_SPY")
    spx_raw_dips = _raw_dips("yfinance_SPX")
    obb_raw_dips = _raw_dips("openbb_SPY")

    if spy_raw_dips is None or spx_raw_dips is None:
        lines.append(
            "SPX vs SPY raw-dip comparison: N/A — one or both sources were "
            "unavailable on this run, so no winner/ties conclusion is drawn."
        )
    elif spx_raw_dips < spy_raw_dips:
        lines.append(
            f"SPX has fewer theta dips ({spx_raw_dips}) than SPY ({spy_raw_dips}) "
            "on raw data, suggesting the non-monotonicity is partially "
            "a SPY-specific data artifact (likely dividend-related noise)."
        )
    elif spx_raw_dips == spy_raw_dips:
        lines.append(
            f"Both SPX ({spx_raw_dips}) and SPY ({spy_raw_dips}) show the same "
            "number of theta dips on raw data. The non-monotonicity is a "
            "genuine market feature, not an underlying-specific artifact."
        )
    else:
        lines.append(
            f"SPX shows more theta dips ({spx_raw_dips}) than SPY ({spy_raw_dips}) "
            "on raw data — unexpected, investigate further."
        )

    if results.get("openbb_SPY") is None:
        lines.append(
            "OpenBB was not available for comparison. Install with "
            "`pip install openbb` to include it in future audits."
        )
    elif spy_raw_dips is None or obb_raw_dips is None:
        lines.append(
            "OpenBB vs yfinance raw-dip comparison: N/A — one or both sources "
            "were unavailable on this run, so no winner/ties conclusion is drawn."
        )
    elif obb_raw_dips < spy_raw_dips:
        lines.append(
            f"OpenBB has fewer theta dips ({obb_raw_dips}) than yfinance "
            f"({spy_raw_dips}), suggesting the OpenBB ingestion path "
            "(normalisation) may help."
        )
    elif obb_raw_dips == spy_raw_dips:
        lines.append(
            f"OpenBB ({obb_raw_dips}) and yfinance ({spy_raw_dips}) show the "
            "same number of theta dips — the ingestion path does not matter "
            "(provider independence is not tested: OpenBB uses provider='yfinance')."
        )

    lines.append("")

    # Replace only this bounded section so later Issue #15 sections,
    # including the determinism snapshot, are preserved.  Accept both the
    # pre-FIX-4 header ("Data source comparison") and the current one so a
    # rerun replaces the old tracked section instead of duplicating it.
    markers = ["### Data source comparison", "### Underlying / path comparison"]
    found_marker = next((m for m in markers if m in content), None)
    if found_marker is not None:
        start_idx = content.index(found_marker)
        # Find the next ### heading after the start marker
        next_section_idx = content.find("\n### ", start_idx + len(found_marker))
        if next_section_idx == -1:
            # No next section — replace from marker to end of file
            new_data_source_section = "\n".join(lines) + "\n"
            content = content[:start_idx].rstrip() + "\n" + new_data_source_section
        else:
            # Replace only the data source comparison section
            new_data_source_section = "\n".join(lines) + "\n"
            content = (
                content[:start_idx].rstrip()
                + "\n"
                + new_data_source_section
                + "\n"
                + content[next_section_idx + 1:]  # keep the \n before next section
            )
    else:
        # No existing section — append
        content = content.rstrip() + "\n" + "\n".join(lines) + "\n"

    issues_path.write_text(content, encoding="utf-8")
    print(f"\n  Wrote underlying/path comparison to {issues_path}")


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    audit_results = run_audit()
    write_findings_to_issues(audit_results)
