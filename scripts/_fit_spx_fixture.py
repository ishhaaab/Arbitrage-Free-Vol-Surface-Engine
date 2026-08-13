"""Run eSSVI fit on a saved SPX fixture and print the fallback list.

This is an internal-use utility for determinism checks; it is not part of the
regular application workflow.

Used for determinism testing — run this in separate process invocations
on the SAME fixture and compare the fallback lists.
"""
import sys
import json
from pathlib import Path
from math import log

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.forward import estimate_forward_curve, populate_per_slice_r
from arbfree_vol.ssvi.term_structure import fit_ssvi_surface_sequential


def load_fixture(path: Path) -> VolSurface:
    """Load a JSON fixture back into a VolSurface."""
    data = json.loads(path.read_text(encoding="utf-8"))

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
        sl = ExpirySlice(
            expiry_time=sl_data["expiry_time"],
            risk_free=sl_data.get("risk_free"),
            div_yield=sl_data.get("div_yield"),
            quotes=quotes,
        )
        slices.append(sl)

    return VolSurface(
        spot=data["spot"],
        risk_free=data["risk_free"],
        div_yield=data["div_yield"],
        slices=slices,
    )


def extract_slice_data(surface):
    """Extract (T, [(k, w)]) from a VolSurface for the eSSVI fit."""
    fwd_curve = estimate_forward_curve(surface)
    populate_per_slice_r(surface, fwd_curve)

    from arbfree_vol.variance import slice_total_variance

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


def main():
    fixture_path = Path(__file__).resolve().parent / "_fixtures" / "spx_raw.json"
    if not fixture_path.exists():
        print(
            f"Fixture not found: {fixture_path}\n"
            "Run scripts/_save_spx_fixture.py first to capture the SPX snapshot.",
            file=sys.stderr,
        )
        sys.exit(1)
    print(f"Loading fixture from {fixture_path}")
    surface = load_fixture(fixture_path)
    print(f"Loaded {len(surface.slices)} slices, spot={surface.spot:.2f}")

    slices_data = extract_slice_data(surface)
    print(f"Extracted {len(slices_data)} slices for fitting")

    result = fit_ssvi_surface_sequential(slices_data)

    print(f"\nFitted: {len(result.fitted_slices)}")
    print(f"Fallback: {len(result.fallback_slices)}")
    print(f"Failed: {len(result.failed_slices)}")
    print("\nFallback T values (in order):")
    for T in result.fallback_slices:
        print(f"  {T:.6f}")
    print("\nFailed T values (in order):")
    for T in result.failed_slices:
        print(f"  {T:.6f}")


if __name__ == "__main__":
    main()
