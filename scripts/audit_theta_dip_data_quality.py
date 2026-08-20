"""Audit: data quality at eSSVI fallback vs non-fallback expiries.

Thin driver for the audit library (``arbfree_vol.data.audit``) — all
computation, report rendering, and the Issue #15 findings writer live
there and are unit-tested.  This script owns only the CLI.

RESEARCH / DIAGNOSTIC tool — not part of the library test suite.  All
numbers are snapshot-in-time: they come from live fetches on the day
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

Usage:
    python scripts/audit_theta_dip_data_quality.py
    python scripts/audit_theta_dip_data_quality.py --use-fixture
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arbfree_vol.data.audit import (  # noqa: E402
    run_audit,
    write_findings_to_issues,
)

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


def main() -> int:
    args = parse_args()
    results = run_audit(args)
    write_findings_to_issues(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())