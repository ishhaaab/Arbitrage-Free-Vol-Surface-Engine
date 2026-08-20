"""Diagnostic: investigate why SPY eSSVI slices take the fallback path.

Thin driver for the diagnostics library (``arbfree_vol.ssvi.diagnostics``)
— all computation, fit attempts, and report rendering live there and are
unit-tested.  This script owns only the CLI.

RESEARCH / DIAGNOSTIC tool — not part of the library test suite.  All
numbers are snapshot- and optimizer-dependent: data is fetched live
(yfinance) at run time, or built synthetically when the fetch fails, and
the hard-constrained fits depend on the scipy optimizer (trust-constr
with an SLSQP retry), its tolerances, and the starting point.

Usage:
    python scripts/diagnose_fallback_slices.py
"""

import logging
import sys
from pathlib import Path

# Ensure project root on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arbfree_vol.ssvi.diagnostics import run_diagnostics  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def main() -> int:
    """Run the fallback-slice diagnostic and report its summary rows."""
    run_diagnostics()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())