"""Ground-truth calibration-recovery tests.

For every calibration case in ``calibration_cases.py``:

- generate the (k, w) surface from the TRUE parameters (fixed-seed noise
  where documented);
- run the repo's calibrator;
- assert each recovered parameter is within the documented tolerance.

The tolerances are deliberately tight enough that a SILENT-DEFAULT failure
mode (old finding 3.6: SABR calibration returning hardcoded
``SABRParams(alpha=0.2, rho=0.0, nu=0.3)``, or any calibrator returning its
initial-guess seed instead of a fit) would blow them by a wide margin — see
``CalibrationCase.default_failure_proof`` on each case.
"""

from __future__ import annotations

import pytest
from pytest import approx

from arbfree_vol.sabr.calibration import calibrate_sabr
from arbfree_vol.ssvi.calibration import fit_ssvi_slice
from arbfree_vol.svi.calibration import calibrate

from tests.ground_truth.calibration_cases import (
    ALL_CALIBRATION_CASES,
    ESSVI_SLICE_RECOVERY,
    SABR_CLEAN_RECOVERY,
    SVI_CLEAN_RECOVERY,
    SVI_NOISY_RECOVERY,
)


def _assert_recovered(name: str, recovered: object, truth: object,
                      tolerances: dict[str, float]) -> None:
    """Assert every parameter of ``recovered`` is within its tolerance."""
    for param, tol in tolerances.items():
        got = float(getattr(recovered, param))
        want = float(getattr(truth, param))
        assert got == approx(want, abs=tol), (
            f"{name}: recovered {param}={got} vs true {want} — outside the "
            f"documented tolerance {tol}.  A silent-default / seed-return "
            f"failure would land far outside these bounds."
        )


@pytest.mark.slow
def test_svi_clean_recovery() -> None:
    """Unconstrained ``calibrate`` recovers far-from-seed SVI truth.

    The true params are deliberately far from ``calibrate``'s default seed
    (b 5x, rho opposite sign, m shifted, sigma 3x): if the calibrator
    silently returned its seed, every tolerance below would fail by a wide
    margin (see ``SVI_CLEAN_RECOVERY.default_failure_proof``).
    """
    case = SVI_CLEAN_RECOVERY
    fit = calibrate(case.points())
    _assert_recovered(case.name, fit, case.true_params, case.tolerances)


def test_essvi_slice_recovery() -> None:
    """``fit_ssvi_slice`` recovers the eSSVI slice truth (fast comfort case)."""
    case = ESSVI_SLICE_RECOVERY
    fit = fit_ssvi_slice(case.points())
    _assert_recovered(case.name, fit, case.true_params, case.tolerances)


@pytest.mark.slow
def test_sabr_clean_recovery_guards_finding_36() -> None:
    """``calibrate_sabr`` recovers truth; the old 3.6 silent default fails.

    Old finding 3.6: failed SABR calibration returned hardcoded
    ``SABRParams(alpha=0.2, rho=0.0, nu=0.3)`` with only a WARNING.  The
    true params here (alpha=0.35, rho=-0.6, nu=0.8) sit far from those
    constants, so a returned default would violate the documented
    tolerances by 20x-120x (``default_failure_proof``).
    """
    case = SABR_CLEAN_RECOVERY
    fit = calibrate_sabr(case.points(), forward=100.0, expiry_time=0.5,
                         beta_hint=0.5)
    _assert_recovered(case.name, fit, case.true_params, case.tolerances)

    # Belt and braces: quantify the margin by which the old default fails.
    default = type(case.true_params)(alpha=0.2, beta=0.5, rho=0.0, nu=0.3)
    assert abs(default.alpha - case.true_params.alpha) > 20 * case.tolerances["alpha"]
    assert abs(default.rho - case.true_params.rho) > 10 * case.tolerances["rho"]
    assert abs(default.nu - case.true_params.nu) > 5 * case.tolerances["nu"]


@pytest.mark.slow
def test_svi_noisy_recovery_documents_achieved_error() -> None:
    """Noisy recovery (fixed seed) meets its documented tolerances.

    The noise is additive on total variance with std 0.001 (~1.4% of the ATM
    total variance) and a FIXED ``default_rng`` seed, so the run is fully
    deterministic.  The tolerances below are the documented recovery bounds;
    the achieved errors with this seed are ~a:1e-3, b:2.4e-3, rho:1e-2,
    m:5e-3, sigma:4.4e-3 — comfortably inside them, and orders of magnitude
    below what a silent-default failure would produce.
    """
    case = SVI_NOISY_RECOVERY
    fit = calibrate(case.points())
    _assert_recovered(case.name, fit, case.true_params, case.tolerances)


@pytest.mark.slow
def test_calibration_cases_have_documented_failure_proofs() -> None:
    """Every calibration case documents how a silent-default failure fails."""
    for case in ALL_CALIBRATION_CASES:
        assert case.default_failure_proof, (
            f"{case.name}: missing default_failure_proof (3.6-style guard)"
        )
        assert case.tolerances, f"{case.name}: missing tolerances"
