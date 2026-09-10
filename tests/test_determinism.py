"""Cross-process determinism checks on the public frozen snapshot."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CSV = _ROOT / "data" / "spx_2026-07-31.csv"
_METADATA = _ROOT / "data" / "spx_2026-07-31_metadata.json"

_CHILD = """
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, __SRC__)
from arbfree_vol.calibration import calibrate_surface
from arbfree_vol.ingestion.loader import load_chain_csv

metadata = json.loads(Path(__METADATA__).read_text(encoding="utf-8"))
surface, rejected = load_chain_csv(
    __CSV__,
    spot=metadata["spot"],
    as_of=date.fromisoformat(metadata["as_of"]),
    risk_free=metadata["risk_free"],
    div_yield=metadata["dividend_yield"],
)
report = calibrate_surface(surface, rejected=rejected)
print(json.dumps({
    "fallback": report.fallback_slices,
    "failed": report.failed_slices,
    "params": [
        [item.expiry_time, item.ssvi.theta, item.ssvi.rho, item.ssvi.psi]
        for item in report.fitted_ssvi_slices
    ],
    "certificate": {
        "variance": report.certificate.min_total_variance,
        "density": report.certificate.min_butterfly_density,
        "calendar": report.certificate.min_calendar_margin,
        "certified": report.certificate.certified,
    },
}))
""".replace("__SRC__", repr(str(_ROOT / "src"))).replace(
    "__METADATA__", repr(str(_METADATA))
).replace("__CSV__", repr(str(_CSV)))


def _run() -> dict:
    result = subprocess.run(
        [sys.executable, "-c", _CHILD],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        pytest.fail(result.stderr)
    return json.loads(result.stdout.strip())


@pytest.mark.slow
def test_frozen_snapshot_failure_state_is_deterministic() -> None:
    first, second = _run(), _run()
    assert first["fallback"] == second["fallback"]
    assert first["failed"] == second["failed"]
    assert first["certificate"] == second["certificate"]


@pytest.mark.slow
def test_frozen_snapshot_parameters_are_deterministic() -> None:
    assert _run()["params"] == _run()["params"]
