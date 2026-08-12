"""Hand-derived calibration-recovery ground-truth cases.

Each case fixes TRUE model parameters, generates a total-variance surface
over a documented strike grid (with optional fixed-seed noise), and records
the recovery tolerances the repo's calibrators must meet.  The tolerances
are deliberately tight enough that a SILENT-DEFAULT failure mode (old
finding 3.6: SABR calibration returning hardcoded alpha=0.2 / rho=0.0 /
nu=0.3, or any calibrator returning its initial-guess seed instead of a
fit) would blow them by a wide margin.

Noise, where present, uses a fixed ``np.random.default_rng`` seed so the
case is fully deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from arbfree_vol.sabr.model import SABRParams
from arbfree_vol.ssvi.model import SSVIParams, ssvi_w
from arbfree_vol.svi.model import SVIParams, svi_total_variance


@dataclass(frozen=True)
class CalibrationCase:
    """True parameters + surface generation + documented recovery bounds.

    Parameters
    ----------
    name : str
        Unique case name.
    model : {"svi", "essvi", "sabr"}
    true_params : SVIParams | SSVIParams | SABRParams
        The ground-truth parameters the surface is generated from.
    k_grid : np.ndarray
        Log-moneyness grid of the synthetic (k, w) points.
    noise_std : float
        Standard deviation of additive total-variance noise, or 0.0 for
        clean data.  ``seed`` fixes the draw.
    seed : int
        Fixed numpy default_rng seed (noise_std > 0 only).
    tolerances : dict[str, float]
        Parameter-name -> absolute tolerance on |recovered - true|.
    default_failure_proof : str
        Arithmetic showing a silent-default / seed-return failure would
        violate the tolerances by a wide margin.
    """

    name: str
    model: str
    true_params: object
    k_grid: np.ndarray = field(repr=False)
    noise_std: float = 0.0
    seed: int = 0
    tolerances: dict[str, float] = field(default_factory=dict)
    default_failure_proof: str = ""

    def points(self) -> list[tuple[float, float]]:
        """Generate the (k, w) points for this case (deterministic)."""
        rng = np.random.default_rng(self.seed)
        pts: list[tuple[float, float]] = []
        for k in self.k_grid:
            w = self._w(float(k))
            if self.noise_std > 0.0:
                w = w + float(rng.normal(0.0, self.noise_std))
            pts.append((float(k), w))
        return pts

    def _w(self, k: float) -> float:
        p = self.true_params
        if self.model == "svi":
            return svi_total_variance(k, p.a, p.b, p.rho, p.m, p.sigma)  # type: ignore[union-attr]
        if self.model == "essvi":
            return ssvi_w(k, p.theta, p.rho, p.psi)  # type: ignore[union-attr]
        if self.model == "sabr":
            return self.sabr_total_variance(k, p)  # type: ignore[arg-type]
        raise ValueError(f"unknown model {self.model}")

    @staticmethod
    def sabr_total_variance(k: float, p: SABRParams) -> float:
        """SABR total variance at the case's forward/expiry (Hagan et al.)."""
        from arbfree_vol.sabr.model import sabr_total_variance as _tv

        return _tv(k, 100.0, 0.5, p.alpha, p.beta, p.rho, p.nu)


# ---------------------------------------------------------------------------
# Case C1 — SVI clean recovery (far from the default seed)
# ---------------------------------------------------------------------------
# True params are deliberately far from ``calibrate``'s default seed
# (a=min(w), b=0.1, rho=-0.5, m=0.0, sigma=0.1): b is 5x the seed's, rho has
# the OPPOSITE sign, m is -0.5 not 0, sigma is 3x the seed's.  If the
# calibrator silently returned its seed (a 3.6-style silent-default failure
# mode), the tolerance checks below would fail by wide margins.
SVI_CLEAN_RECOVERY = CalibrationCase(
    name="svi_clean_recovery",
    model="svi",
    true_params=SVIParams(a=0.05, b=0.5, rho=0.5, m=-0.5, sigma=0.3),
    k_grid=np.linspace(-0.5, 0.5, 13),
    noise_std=0.0,
    seed=0,
    tolerances={"a": 1e-3, "b": 1e-3, "rho": 1e-3, "m": 1e-3, "sigma": 1e-3},
    default_failure_proof=(
        "a silent seed-return (b=0.1, rho=-0.5, m=0.0, sigma=0.1) would "
        "give |b-0.5|=0.4 >> 1e-3, |rho-0.5|=1.0 >> 1e-3, |m+0.5|=0.5 >> "
        "1e-3, |sigma-0.3|=0.2 >> 1e-3"
    ),
)

# ---------------------------------------------------------------------------
# Case C2 — eSSVI slice recovery
# ---------------------------------------------------------------------------
ESSVI_SLICE_RECOVERY = CalibrationCase(
    name="essvi_slice_recovery",
    model="essvi",
    true_params=SSVIParams(theta=0.1, rho=-0.3, psi=1.0),
    k_grid=np.linspace(-0.5, 0.5, 13),
    noise_std=0.0,
    seed=0,
    tolerances={"theta": 1e-3, "rho": 1e-3, "psi": 1e-3},
    default_failure_proof=(
        "a silent seed-return (theta=min w, rho=0.0, psi=0.5) would give "
        "|rho+0.3|=0.3 >> 1e-3 and |psi-1.0|=0.5 >> 1e-3"
    ),
)

# ---------------------------------------------------------------------------
# Case C3 — SABR clean recovery (guard for old finding 3.6)
# ---------------------------------------------------------------------------
# Old finding 3.6: failed SABR calibration returned hardcoded
# SABRParams(alpha=0.2, rho=0.0, nu=0.3) with only a WARNING.  The true
# params here sit far from those constants (alpha 0.35, rho -0.6, nu 0.8),
# so a returned default would fail the documented tolerances by a wide
# margin: |alpha-0.35|=0.15 vs rel 2% (~0.007), |rho+0.6|=0.6 vs 0.05,
# |nu-0.8|=0.5 vs rel 10% (0.08).
SABR_CLEAN_RECOVERY = CalibrationCase(
    name="sabr_clean_recovery",
    model="sabr",
    true_params=SABRParams(alpha=0.35, beta=0.5, rho=-0.6, nu=0.8),
    k_grid=np.linspace(-0.5, 0.5, 13),
    noise_std=0.0,
    seed=0,
    tolerances={"alpha": 0.007, "rho": 0.05, "nu": 0.08},
    default_failure_proof=(
        "old finding 3.6 silent default (alpha=0.2, rho=0.0, nu=0.3) "
        "would give |alpha-0.35|=0.15 >> 0.007, |rho+0.6|=0.6 >> 0.05, "
        "|nu-0.8|=0.5 >> 0.08"
    ),
)

# ---------------------------------------------------------------------------
# Case C4 — SVI noisy recovery at a realistic noise level
# ---------------------------------------------------------------------------
# Total-variance noise std 0.001 at an ATM total variance w(0) ~ 0.07 is a
# ~1.4% relative noise level (a realistic one-tick-ish market perturbation).
# The recovery tolerances are looser than the clean case and the ACHIEVED
# errors are documented by the test (a ~0.001, b ~0.002, rho ~0.010,
# m ~0.005, sigma ~0.005 with this fixed seed).
SVI_NOISY_RECOVERY = CalibrationCase(
    name="svi_noisy_recovery",
    model="svi",
    true_params=SVIParams(a=0.04, b=0.4, rho=-0.4, m=0.05, sigma=0.15),
    k_grid=np.linspace(-0.5, 0.5, 15),
    noise_std=0.001,
    seed=20260812,
    tolerances={"a": 0.01, "b": 0.02, "rho": 0.05, "m": 0.02, "sigma": 0.02},
    default_failure_proof=(
        "noise std 0.001 on w ~ 0.07 is ~1.4% of ATM; a silent seed-return "
        "would give |b-0.4|=0.3, |rho+0.4|=0.8, |sigma-0.15|=0.05 — all far "
        "outside the documented recovery tolerances"
    ),
)

ALL_CALIBRATION_CASES: list[CalibrationCase] = [
    SVI_CLEAN_RECOVERY,
    ESSVI_SLICE_RECOVERY,
    SABR_CLEAN_RECOVERY,
    SVI_NOISY_RECOVERY,
]
