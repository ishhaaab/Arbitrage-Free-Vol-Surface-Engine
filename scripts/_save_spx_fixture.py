"""Save SPX raw option chain data to a fixture file for determinism testing.

This is an internal-use utility for creating audit fixtures; it is not part
of the regular application workflow.

Fetches SPX data ONCE from yfinance and saves the raw quotes/strikes/spot
to a JSON fixture.  The fixture is used for determinism testing — the
fitting pipeline should produce identical results when run on this
fixture multiple times.
"""
import sys
import json
import math
from pathlib import Path
from datetime import date

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arbfree_vol.ingestion.yfinance import fetch_chain


def _sanitize(obj):
    """Recursively replace NaN/Inf with None so JSON can serialize."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize(v) for v in obj]
    return obj


def main():
    print("Fetching SPX raw data from yfinance...")
    surface, rejected, quality_drops = fetch_chain(
        "^SPX",
        max_expiries=40,
        min_T_years=7.0 / 365.0,
        disable_quality_filter=True,  # raw data, no quality filter
    )

    fixture = {
        "spot": surface.spot,
        "risk_free": surface.risk_free,
        "div_yield": surface.div_yield,
        "ref_date": str(date.today()),
        "slices": [],
    }

    for sl in surface.slices:
        slice_data = {
            "expiry_time": sl.expiry_time,
            "risk_free": sl.risk_free,
            "div_yield": sl.div_yield,
            "quotes": [
                {
                    "strike": q.strike,
                    "option_type": q.option_type.value,
                    "price": q.price,
                    "bid": q.bid,
                    "ask": q.ask,
                }
                for q in sl.quotes
            ],
        }
        fixture["slices"].append(slice_data)

    fixture = _sanitize(fixture)

    fixture_path = Path(__file__).resolve().parent / "_fixtures" / "spx_raw.json"
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    fixture_path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")

    print(f"Saved {len(fixture['slices'])} slices to {fixture_path}")
    print(f"Spot: {fixture['spot']}")
    print(f"Total quotes: {sum(len(s['quotes']) for s in fixture['slices'])}")


if __name__ == "__main__":
    main()
