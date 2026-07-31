"""Determinism regression test for the eSSVI sequential fit.

Verifies that the fitting pipeline produces identical output when run
twice on the same saved fixture in separate process invocations.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "spx_sample.json"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CHILD_SCRIPT = """
import json
import sys
from math import log
from pathlib import Path

sys.path.insert(0, __PROJECT_ROOT__)

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.repair.fwd_curve import estimate_forward_curve, populate_per_slice_r
from arbfree_vol.ssvi.term_structure import fit_ssvi_surface_sequential
from arbfree_vol.variance import slice_total_variance

data = json.loads(Path(__FIXTURE_PATH__).read_text(encoding="utf-8"))
slices = []
for sl_data in data["slices"]:
    quotes = [
        Quote(
            strike=q["strike"],
            option_type=OptionType(q["option_type"]),
            price=q["price"],
            bid=q.get("bid"),
            ask=q.get("ask"),
        )
        for q in sl_data["quotes"]
        if q["price"] is not None
    ]
    if quotes:
        slices.append(ExpirySlice(
            expiry_time=sl_data["expiry_time"],
            risk_free=sl_data.get("risk_free"),
            div_yield=sl_data.get("div_yield"),
            quotes=quotes,
        ))

surface = VolSurface(
    spot=data["spot"],
    risk_free=data["risk_free"],
    div_yield=data["div_yield"],
    slices=slices,
)
fwd_curve = estimate_forward_curve(surface)
populate_per_slice_r(surface, fwd_curve)
slices_data = []
for sl in sorted(surface.slices, key=lambda s: s.expiry_time):
    forward = fwd_curve.get(sl.expiry_time)
    if forward is None:
        continue
    strike_w = slice_total_variance(surface, sl)
    if len(strike_w) < 5:
        continue
    points = [(log(strike / forward), w) for strike, w in strike_w.items()]
    points.sort()
    slices_data.append((sl.expiry_time, points))

result = fit_ssvi_surface_sequential(slices_data)
print(json.dumps({
    "fallback_slices": result.fallback_slices,
    "failed_slices": result.failed_slices,
    "fitted_slices": [
        {"T": T, "theta": p.theta, "rho": p.rho, "psi": p.psi}
        for T, p in result.fitted_slices
    ],
}))
""".replace(
    "__PROJECT_ROOT__", repr(str(_PROJECT_ROOT))
).replace(
    "__FIXTURE_PATH__", repr(str(_FIXTURE_PATH))
)


def _run_fit_subprocess() -> dict:
    """Run the fit in a fresh Python subprocess and return parsed JSON."""
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=_PROJECT_ROOT,
    )
    if result.returncode != 0:
        pytest.fail(f"Child process failed: {result.stderr}")
    return json.loads(result.stdout.strip())


def test_determinism_fallback_lists_match() -> None:
    """Identical input must produce identical fallback and failed lists."""
    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Fixture not found: {_FIXTURE_PATH}")

    run1 = _run_fit_subprocess()
    run2 = _run_fit_subprocess()
    assert run1["fallback_slices"] == run2["fallback_slices"]
    assert run1["failed_slices"] == run2["failed_slices"]


def test_determinism_fitted_params_match() -> None:
    """Identical input must produce identical fitted parameters."""
    if not _FIXTURE_PATH.exists():
        pytest.skip(f"Fixture not found: {_FIXTURE_PATH}")

    run1 = _run_fit_subprocess()
    run2 = _run_fit_subprocess()
    assert run1["fitted_slices"] == run2["fitted_slices"]
