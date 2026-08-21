"""CLI demo: walk through ``arbfree detect|repair|price``.

Builds a small synthetic option chain CSV, then runs the ``arbfree``
CLI against it and prints the outputs, so you can see the whole
toolkit without a network or interactive prompts.

Usage::

    python demo/cli/cli_demo.py                 # synthetic chain, detect/repair/price
    python demo/cli/cli_demo.py --keep          # keep the generated CSV + report
"""

import argparse
import csv
import subprocess
import sys
from datetime import date, timedelta
from math import sqrt
from pathlib import Path

import numpy as np

_OUT = Path(__file__).parent
_CHAIN_CSV = _OUT / "demo_chain.csv"
_REPORT_JSON = _OUT / "demo_report.json"

parser = argparse.ArgumentParser(description="arbfree CLI walkthrough demo")
parser.add_argument("--keep", action="store_true",
                    help="Keep generated CSV/report after the demo.")
args = parser.parse_args()

AS_OF = date(2030, 7, 15)


def run(cmd: list[str]) -> None:
    """Run a CLI command, echo it, and print its stdout."""
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.stdout:
        print(res.stdout.rstrip())
    if res.stderr:
        print(res.stderr.rstrip(), file=sys.stderr)
    if res.returncode != 0:
        print(f"[exit {res.returncode}]")
        sys.exit(1)


def build_chain_csv() -> None:
    """Write a clean 2-expiry SVI chain (with bid/ask so cleaning passes)."""
    from arbfree_vol.models.option import OptionType
    from arbfree_vol.svi.model import SVIParams, svi_total_variance
    from arbfree_vol.pricing.black_scholes import price_floats

    spot, r, q = 100.0, 0.05, 0.01
    params = {
        0.25: SVIParams(a=0.010, b=0.32, rho=-0.35, m=0.0, sigma=0.11),
        1.0:  SVIParams(a=0.035, b=0.40, rho=-0.42, m=0.04, sigma=0.16),
    }
    rows = []
    for T, p in params.items():
        exp = AS_OF + timedelta(days=int(round(T * 365)))
        for k in np.linspace(-0.30, 0.30, 7):
            K = spot * np.exp(k)
            w = svi_total_variance(float(k), p.a, p.b, p.rho, p.m, p.sigma)
            sigma = sqrt(w / T)
            for otype in (OptionType.CALL, OptionType.PUT):
                price = price_floats(spot, K, T, r, q, sigma,
                                     is_call=(otype == OptionType.CALL))
                spread = 0.02 * price + 0.01
                rows.append([
                    f"{K:.2f}", exp.isoformat(),
                    "C" if otype == OptionType.CALL else "P",
                    f"{price:.6f}", f"{price - spread / 2:.6f}",
                    f"{price + spread / 2:.6f}",
                ])
    with open(_CHAIN_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["strike", "expiry", "option_type", "price", "bid", "ask"])
        w.writerows(rows)
    print(f"Wrote {_CHAIN_CSV.name}: {len(rows)} quotes "
          f"(strike,expiry,option_type,price,bid,ask) "
          f"across 2 expiries.")


def main() -> None:
    print("=" * 64)
    print("   arbfree CLI walkthrough (synthetic chain, no network)")
    print("=" * 64)

    build_chain_csv()

    run(["python", "-m", "arbfree_vol.cli", "--version"])
    run(["python", "-m", "arbfree_vol.cli", "detect", str(_CHAIN_CSV),
         "--spot", "100", "--as-of", AS_OF.isoformat()])
    run(["python", "-m", "arbfree_vol.cli", "repair", str(_CHAIN_CSV),
         "--spot", "100", "--as-of", AS_OF.isoformat(),
         "--use-ssvi", "--output", str(_REPORT_JSON)])
    run(["python", "-m", "arbfree_vol.cli", "price",
         "--spot", "100", "--strike", "100", "--expiry", "0.25",
         "--vol", "0.2"])
    run(["python", "-m", "arbfree_vol.cli", "price",
         "--spot", "100", "--strike", "100", "--expiry", "0.25",
         "--price", "4.61", "--div-yield", "0.01"])

    print("\n" + "=" * 64)
    print("   CLI demo complete.")
    print("=" * 64)
    print("\nWhat you just saw:")
    print("  detect  - the 5 static arbitrage checks on a CSV chain")
    print("  repair  - clean + fit + re-validate; writes a JSON report")
    print("  price   - Black-Scholes price (with --vol) and implied vol")
    print("            (with --price)")

    if not args.keep:
        _CHAIN_CSV.unlink(missing_ok=True)
        _REPORT_JSON.unlink(missing_ok=True)
        print("\n(Removed demo_chain.csv and demo_report.json; "
              "use --keep to retain them.)")


if __name__ == "__main__":
    main()
