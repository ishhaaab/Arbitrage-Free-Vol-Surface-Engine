"""Run eSSVI fit on the saved SPX fixture and print the fallback list.

Internal determinism-check utility; not part of the regular application
workflow.  Run this in separate process invocations on the SAME fixture
and compare the fallback lists (the pipeline must be deterministic).

The committed fixture is ``tests/fixtures/spx_sample.json`` (7-slice
post-ingestion ^SPX snapshot).  NOTE: this fixture currently fits with
ZERO fallbacks, so it exercises determinism only -- use live fetches or
the synthetic diagnostics fixture (``arbfree_vol.ssvi.diagnostics``) to
reproduce fallback behavior.
"""
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from arbfree_vol.data.audit import load_spx_fixture  # noqa: E402
from arbfree_vol.data.audit import extract_slice_data  # noqa: E402
from arbfree_vol.ssvi.term_structure import fit_ssvi_surface_sequential  # noqa: E402


def main():
    surface = load_spx_fixture()
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
