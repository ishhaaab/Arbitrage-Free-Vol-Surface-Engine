"""Hand-derived arbitrage ground-truth cases (eSSVI and raw SVI).

Every label in this module is derived by hand from the analytic conditions:

- Gatheral & Jacquier (2014) Theorem 4.2 (sufficiency) and Lemma 4.2
  (necessity / tightness) for the per-slice butterfly bounds:

      theta * psi * (1 + |rho|) < 4        (condition 1, STRICT)
      theta * psi^2 * (1 + |rho|) <= 4     (condition 2, non-strict)

  Lemma 4.2 is load-bearing for the "second-bound-only" and "boundary"
  cases: condition 1 is NECESSARY (violating it is a genuine butterfly
  violation), while condition 2 is only SUFFICIENT unless condition 1 sits
  exactly at its boundary — in which case condition 2 becomes necessary
  too.  That is why the second-bound-only construction below is placed at
  condition 1 == 4 exactly: only there does violating condition 2
  guarantee a genuine (grid-detectable) violation.

- Hendriks & Martini (2019) Prop 3.1 for the calendar conditions that
  ``verify_hm_condition`` checks (theta non-decreasing, chi = theta*psi
  non-decreasing, |rho*chi| ratio <= 1) and the native-SSVI grid check
  ``verify_ssvi_calendar_free`` (w_{i+1}(k) >= w_i(k) on a dense grid).

- Gatheral (2004) / SVI density ``g(k)`` (the repo's ``svi_g`` formula)
  implemented LITERALLY below for the SVI cases, so the raw-SVI labels do
  not depend on the repo's ``svi_g`` implementation.

All cases are self-derived; no label calls any repo detector or verifier.
"""

from __future__ import annotations

from math import sqrt

from arbfree_vol.sabr.model import SABRParams  # noqa: F401 (schema completeness)
from arbfree_vol.ssvi.model import (
    SSVIParams,
    eSSVISurfaceParams,
    to_raw_svi_params,
)
from arbfree_vol.svi.model import SVIParams

from tests.ground_truth.cases import GroundTruthCase


# ---------------------------------------------------------------------------
# Independent literal implementations (anti-circularity: these are the
# hand-derived formulas, not imports of the repo's helpers).
# ---------------------------------------------------------------------------

def gj_first_bound(theta: float, rho: float, psi: float) -> float:
    """GJ 2014 Theorem 4.2 condition 1 lhs: theta*psi*(1+|rho|)."""
    return theta * psi * (1.0 + abs(rho))


def gj_second_bound(theta: float, rho: float, psi: float) -> float:
    """GJ 2014 Theorem 4.2 condition 2 lhs: theta*psi^2*(1+|rho|)."""
    return theta * psi * psi * (1.0 + abs(rho))


# Canonical strictness epsilon for GJ condition 1 (the repo's
# ``_GJ_STRICT_EPS`` semantics, restated literally here).
_GJ_STRICT_EPS: float = 1e-9


def gj_hand_residual(theta: float, rho: float, psi: float,
                     strict: bool = False) -> float:
    """Hand-computed Gatheral-Jacquier residual ``min(4-b1, 4-b2)``.

    ``strict=True`` applies the paper's STRICT condition 1 reading: an exact
    equality on condition 1 (residual within ``_GJ_STRICT_EPS`` of 0) is
    nudged negative so the boundary is never reported safe — mirroring the
    repo's documented strict-mode semantics without calling it.
    """
    r1 = 4.0 - gj_first_bound(theta, rho, psi)
    r2 = 4.0 - gj_second_bound(theta, rho, psi)
    if strict and r1 <= _GJ_STRICT_EPS:
        r1 = r1 - _GJ_STRICT_EPS
    return min(r1, r2)


def svi_density_g(k: float, a: float, b: float, rho: float, m: float,
                  sigma: float) -> float:
    """Independent literal implementation of Gatheral's g(k) density.

    ``g(k) = (1 - k*w1/(2*w0))^2 - (w1^2/4)*(1/w0 + 1/4) + w2/2`` with
    ``w0, w1, w2`` the SVI total variance and its first/second k-derivatives
    (the repo's ``svi_core``).  Returns ``-inf`` when w0 <= 0 (no density
    exists), mirroring the repo's convention.
    """
    u = k - m
    r = sqrt(u * u + sigma * sigma)
    w0 = a + b * (rho * u + r)
    if w0 <= 0.0:
        return float("-inf")
    w1 = b * (rho + u / r)
    w2 = b * sigma * sigma / (r ** 3)
    return ((1.0 - k * w1 / (2.0 * w0)) ** 2
            - (w1 * w1 / 4.0) * (1.0 / w0 + 1.0 / 4.0)
            + w2 / 2.0)


def dense_g_min(a: float, b: float, rho: float, m: float, sigma: float,
                k_lo: float = -3.0, k_hi: float = 3.0,
                n: int = 4001) -> tuple[float, float]:
    """Dense-grid minimum of the independent g(k) and its location."""
    best_k, best_g = 0.0, float("inf")
    for i in range(n):
        k = k_lo + (k_hi - k_lo) * i / (n - 1)
        g = svi_density_g(k, a, b, rho, m, sigma)
        if g < best_g:
            best_g, best_k = g, k
    return best_k, best_g


# ---------------------------------------------------------------------------
# Case construction helpers
# ---------------------------------------------------------------------------

def _surface(
    name: str,
    slices: list[SSVIParams],
    label: str,
    hm: bool | None,
    grid: bool | None,
    proof_note: str,
    *,
    eta: float | None = None,
    gamma: float | None = None,
) -> GroundTruthCase:
    """Build an eSSVI surface case from per-slice params."""
    surface = eSSVISurfaceParams(eta=eta, gamma=gamma) if eta is not None else None
    return GroundTruthCase(
        name=name,
        model="essvi",
        params=surface,
        slices=tuple(slices),
        known_label=label,  # type: ignore[arg-type]
        expected_hm_condition=hm,
        expected_grid_calendar_free=grid,
        source="self-derived (GJ 2014 Theorem 4.2 + Lemma 4.2 conditions; "
               "HM 2019 Prop 3.1 conditions)",
        proof_note=proof_note,
    )


# ---------------------------------------------------------------------------
# Case 1 — eSSVI ARB-FREE interior
# ---------------------------------------------------------------------------
# Hand arithmetic (all three slices, rho fixed at -0.25):
#   slice 1: theta*psi*(1+|rho|) = 0.1*1.0*1.25 = 0.125 < 4  (margin 3.875)
#            theta*psi^2*(1+|rho|) = 0.125 < 4
#   slice 2: 0.2*1.0*1.25 = 0.25 < 4;  second = 0.25 < 4
#   slice 3: 0.4*1.0*1.25 = 0.5 < 4;   second = 0.5 < 4
# Calendar (HM Prop 3.1): theta 0.1 -> 0.2 -> 0.4 non-decreasing;
#   chi = theta*psi = 0.1 -> 0.2 -> 0.4 non-decreasing;
#   ratio = |rho*(chi_{i+1} - chi_i)| / (chi_{i+1} - chi_i) = |rho| = 0.25 <= 1.
ESSVI_ARB_FREE = _surface(
    "essvi_arb_free_interior",
    [
        SSVIParams(theta=0.1, rho=-0.25, psi=1.0),
        SSVIParams(theta=0.2, rho=-0.25, psi=1.0),
        SSVIParams(theta=0.4, rho=-0.25, psi=1.0),
    ],
    label="arb_free",
    hm=True,
    grid=True,
    proof_note=(
        "first bound = theta*psi*(1+|rho|) = 0.4*1.0*1.25 = 0.5 < 4; "
        "second bound = theta*psi^2*(1+|rho|) = 0.4*1.0*1.25 = 0.5 < 4 "
        "(slices 0.125/0.25); theta 0.1->0.2->0.4 and chi 0.1->0.2->0.4 "
        "non-decreasing, ratio = |rho| = 0.25 <= 1 -> HM holds and the "
        "dense-grid w(k) surface is non-decreasing in T."
    ),
    eta=0.4,
    gamma=0.0,
)

# ---------------------------------------------------------------------------
# Case 2 — eSSVI VIOLATES-FIRST-BOUND-ONLY
# ---------------------------------------------------------------------------
# theta*psi*(1+|rho|) > 4 while theta*psi^2*(1+|rho|) <= 4.  Per GJ 2014
# Lemma 4.2 condition 1 is NECESSARY, so violating it is a genuine butterfly
# violation (the mapped raw-SVI density is negative in the far wings: gmin
# ~ -0.095 at |k| ~ 12, beyond the repo's [-3,3] grid — documented below).
# Calendar conditions still hold (theta and chi increase, rho = 0).
ESSVI_VIOLATES_FIRST_ONLY = _surface(
    "essvi_violates_first_bound_only",
    [
        SSVIParams(theta=0.1, rho=0.0, psi=0.8),
        SSVIParams(theta=0.2, rho=0.0, psi=0.8),
        SSVIParams(theta=6.0, rho=0.0, psi=0.8),
    ],
    label="butterfly_violation",
    hm=True,
    grid=True,
    proof_note=(
        "slice 3: first bound = 6.0*0.8*1.0 = 4.8 > 4 (violated, NECESSARY "
        "per GJ 2014 Lemma 4.2); second bound = 6.0*0.64*1.0 = 3.84 <= 4 "
        "(holds).  Mapped raw-SVI density is negative in the far wings "
        "(gmin ~ -0.095 near |k| = 12); on the repo's [-3,3] grid the min "
        "is +0.11, so the [-3,3] grid detector cannot see this violation — "
        "the analytic bound is load-bearing.  theta/chi non-decreasing, "
        "ratio = 0 -> HM and dense-grid calendar checks pass."
    ),
)

# ---------------------------------------------------------------------------
# Case 3 — eSSVI VIOLATES-SECOND-BOUND-ONLY
# ---------------------------------------------------------------------------
# theta*psi*(1+|rho|) <= 4 while theta*psi^2*(1+|rho|) > 4.  Per GJ 2014
# Lemma 4.2 condition 2 is only SUFFICIENT in general: it becomes necessary
# only when condition 1 sits exactly at its boundary (theta*psi*(1+|rho|) =
# 4).  We therefore place this case at that boundary: theta=2.0, psi=2.0,
# rho=0.0 gives condition 1 = 4.0 EXACTLY (in float) and condition 2 = 8.0
# > 4 — a genuine butterfly violation, and the ONLY second-bound-only regime
# the theorem certifies as violating.
ESSVI_VIOLATES_SECOND_ONLY = _surface(
    "essvi_violates_second_bound_only",
    [
        SSVIParams(theta=0.1, rho=0.0, psi=2.0),
        SSVIParams(theta=0.2, rho=0.0, psi=2.0),
        SSVIParams(theta=2.0, rho=0.0, psi=2.0),
    ],
    label="butterfly_violation",
    hm=True,
    grid=True,
    proof_note=(
        "slice 3: first bound = 2.0*2.0*1.0 = 4.0 exactly (float-exact: "
        "2.0*2.0*1.0 = 4.0); second bound = 2.0*4.0*1.0 = 8.0 > 4.  GJ 2014 "
        "Lemma 4.2: condition 1 = 4 forces condition 2 <= 4 for "
        "butterfly-freeness, so condition 2 = 8 > 4 is a genuine violation "
        "(mapped raw-SVI density gmin ~ -0.033 on [-3,3], located near "
        "|k| ~ 3.8).  theta/chi non-decreasing, ratio = 0 -> HM and "
        "dense-grid calendar checks pass."
    ),
)

# ---------------------------------------------------------------------------
# Case 4 — eSSVI VIOLATES-BOTH
# ---------------------------------------------------------------------------
ESSVI_VIOLATES_BOTH = _surface(
    "essvi_violates_both",
    [
        SSVIParams(theta=0.1, rho=0.0, psi=1.5),
        SSVIParams(theta=0.2, rho=0.0, psi=1.5),
        SSVIParams(theta=3.0, rho=0.0, psi=1.5),
    ],
    label="both_violation",
    hm=True,
    grid=True,
    proof_note=(
        "slice 3: first bound = 3.0*1.5*1.0 = 4.5 > 4; second bound = "
        "3.0*2.25*1.0 = 6.75 > 4 — both GJ bounds violated, mapped raw-SVI "
        "density negative in the wings (gmin ~ -0.085 on [-3,3]).  "
        "theta/chi non-decreasing, ratio = 0 -> HM and dense-grid calendar "
        "checks pass."
    ),
)

# ---------------------------------------------------------------------------
# Case 5 — eSSVI BOUNDARY (feasibility edge, condition 1 exactly 4)
# ---------------------------------------------------------------------------
# theta*psi*(1+|rho|) == 4 EXACTLY in float (theta=2.0, psi=2.0, rho=0.0).
# The paper's Theorem 4.2 condition 1 is STRICT, so an exact equality is NOT
# certified butterfly-safe; Lemma 4.2 makes condition 2 necessary at the
# boundary, and here condition 2 = 8.0 > 4 — the slice genuinely carries
# butterfly arbitrage.  This is the case designed to expose old finding 3.10
# (repair_infeasible derived only from verify_hm_condition): the H&M
# parameter check passes (theta monotone) while the actual/grid violation
# check fails.
ESSVI_BOUNDARY = _surface(
    "essvi_boundary_condition1_exact",
    [
        SSVIParams(theta=0.1, rho=0.0, psi=2.0),
        SSVIParams(theta=0.2, rho=0.0, psi=2.0),
        SSVIParams(theta=2.0, rho=0.0, psi=2.0),
    ],
    label="boundary",
    hm=True,
    grid=True,
    proof_note=(
        "theta*psi*(1+|rho|) = 2.0*2.0*1.0 = 4.0 exactly in float; "
        "theta*psi^2*(1+|rho|) = 2.0*4.0*1.0 = 8.0 > 4.  Condition 1 sits "
        "exactly on the strict Theorem 4.2 boundary (not certified safe) "
        "and Lemma 4.2 then makes condition 2 necessary — 8.0 > 4 is a "
        "genuine violation (mapped raw-SVI gmin ~ -0.033 on [-3,3]).  "
        "theta/chi non-decreasing, ratio = 0 -> HM parameter check passes; "
        "only the grid/actual-violation checks can reveal the violation — "
        "the 3.10 honesty structure."
    ),
)

# ---------------------------------------------------------------------------
# Case 5b — eSSVI BOUNDARY, both bounds exactly 4
# ---------------------------------------------------------------------------
# The task brief suggested "theta=2.0, psi=1.0, rho=0.0" for a both-bounds
# boundary; that arithmetic gives 2.0, NOT 4.0 (verified against the actual
# psi convention in ssvi/model.py — psi enters linearly in condition 1 and
# quadratically in condition 2, so both bounds equal 4 requires psi = 1 and
# theta = 4).  The corrected case is theta=4.0, psi=1.0, rho=0.0: both
# bounds are exactly 4.0.  The density touches zero only asymptotically, so
# the [-3,3] grid detector does NOT flag it (gmin ~ +0.13); the label is
# carried by the strict condition-1 boundary.
ESSVI_BOUNDARY_BOTH = _surface(
    "essvi_boundary_both_bounds_exact",
    [
        SSVIParams(theta=0.1, rho=0.0, psi=1.0),
        SSVIParams(theta=0.5, rho=0.0, psi=1.0),
        SSVIParams(theta=4.0, rho=0.0, psi=1.0),
    ],
    label="boundary",
    hm=True,
    grid=True,
    proof_note=(
        "theta*psi*(1+|rho|) = 4.0*1.0*1.0 = 4.0 exactly; "
        "theta*psi^2*(1+|rho|) = 4.0*1.0*1.0 = 4.0 exactly — both GJ bounds "
        "sit exactly on their boundary (condition 1 is STRICT in Theorem "
        "4.2, so the equality is not certified safe).  The density is "
        "nonnegative with an asymptotic zero (mapped raw-SVI gmin ~ +0.13 "
        "on [-3,3]), so the grid detector cannot flag it; the label rests "
        "on the strict condition-1 boundary.  theta/chi non-decreasing, "
        "ratio = 0 -> HM and dense-grid calendar checks pass."
    ),
)

# ---------------------------------------------------------------------------
# Case 6 — eSSVI CALENDAR VIOLATION (theta dips)
# ---------------------------------------------------------------------------
ESSVI_CALENDAR_VIOLATION = _surface(
    "essvi_calendar_violation_theta_dip",
    [
        SSVIParams(theta=0.14, rho=-0.3, psi=0.5),
        SSVIParams(theta=0.08, rho=-0.2, psi=0.6),
        SSVIParams(theta=0.10, rho=-0.1, psi=0.65),
    ],
    label="calendar_violation",
    hm=False,
    grid=False,
    proof_note=(
        "theta 0.14 -> 0.08 -> 0.10 dips at slice 2: HM condition (a) "
        "(theta non-decreasing) fails, so verify_hm_condition is False.  "
        "chi = theta*psi = 0.07 -> 0.048 -> 0.065 also dips at slice 2 "
        "(condition (b) fails).  The total-variance surface crosses "
        "(w2(k) < w1(k) on part of the dense grid), so the native grid "
        "calendar check is False.  Per-slice butterfly bounds are all far "
        "below 4 (max first bound 0.14*0.5*1.3 = 0.091) — the violation is "
        "purely calendar."
    ),
)

# ---------------------------------------------------------------------------
# Case 7 — eSSVI CALENDAR BOUNDARY (theta flat)
# ---------------------------------------------------------------------------
ESSVI_CALENDAR_BOUNDARY = _surface(
    "essvi_calendar_boundary_theta_flat",
    [
        SSVIParams(theta=0.1, rho=-0.2, psi=0.5),
        SSVIParams(theta=0.1, rho=-0.2, psi=0.5),
        SSVIParams(theta=0.1, rho=-0.2, psi=0.5),
    ],
    label="boundary",
    hm=True,
    grid=True,
    proof_note=(
        "theta flat at 0.1 across all slices: HM condition (a) holds with "
        "theta_delta = 0.0 exactly — the non-decreasing monotonicity "
        "constraint sits exactly on its equality boundary (the feasibility "
        "edge of Prop 3.1).  chi = 0.05 flat (condition (b) equality), "
        "ratio = 0.  Slices are identical so w(k) is identical across T and "
        "the dense-grid calendar check passes (gap 0).  Butterfly bounds "
        "0.1*0.5*1.2 = 0.06 < 4."
    ),
)

# ---------------------------------------------------------------------------
# Case 10 — H&M SUFFICIENCY GAP (calendar crossing with hm passing)
# ---------------------------------------------------------------------------
# The repo's own FIX-6 counterexample (docs/review_campaign.md, Item-2):
# this SSVI pair satisfies ALL of HM Prop 3.1 (verify_hm_condition True)
# yet the slices cross (w2 - w1 = -0.0015283 at k = 0.68).  The native-SSVI
# grid check catches the crossing; under the old finding 3.10 code
# (repair_infeasible derived ONLY from verify_hm_condition) this pair was
# silently certified.  This is the non-vacuous 3.10 regression guard.
ESSVI_HM_INSUFFICIENCY = _surface(
    "essvi_hm_insufficiency_calendar_cross",
    [
        SSVIParams(theta=0.0149505446, rho=-0.6548551, psi=0.11491999),
        SSVIParams(theta=0.0574982989, rho=-0.8830506, psi=2.5226500),
    ],
    label="calendar_violation",
    hm=True,
    grid=False,
    proof_note=(
        "HM Prop 3.1 parameter checks all pass: theta 0.01495 -> 0.05750 "
        "non-decreasing; chi = theta*psi = 0.0017181 -> 0.1450481 "
        "non-decreasing; |rho2*chi2 - rho1*chi1| / (chi2 - chi1) = "
        "|(-0.8830506*0.1450481) - (-0.6548551*0.0017181)| / 0.1433300 = "
        "0.886 <= 1.  YET the slices cross: w2(k) - w1(k) = -0.0015283 at "
        "k = 0.68 (H&M conditions are necessary, not sufficient, for "
        "calendar-freeness).  The native grid check (dense k-grid) returns "
        "False.  This is the repo's documented FIX-6 counterexample; the "
        "repair honesty test asserts repair_infeasible=True even though the "
        "param-level check passes."
    ),
)

# ---------------------------------------------------------------------------
# Case 8 — SVI ARB-FREE, mapped from an arb-free eSSVI slice
# ---------------------------------------------------------------------------
# to_raw_svi_params(theta=0.4, rho=-0.25, psi=1.0):
#   b = theta*psi/2 = 0.2; m = -rho/psi = 0.25;
#   sigma = sqrt(1 - rho^2)/psi = sqrt(0.9375) = 0.9682458366;
#   a = (theta/2)*(1 - rho^2) = 0.1875; rho = -0.25.
# The label is inherited from the eSSVI guarantee (all GJ bounds well below
# 4, so the mapped slice is butterfly-free) and corroborated below by the
# independent literal g(k): min g ~ +0.404 on the dense grid.
SVI_ARB_FREE_FROM_ESSVI = GroundTruthCase(
    name="svi_arb_free_from_essvi",
    model="svi",
    params=SVIParams(a=0.1875, b=0.2, rho=-0.25, m=0.25,
                     sigma=0.9682458365518543),
    known_label="arb_free",
    expected_hm_condition=None,
    expected_grid_calendar_free=None,
    source="self-derived (GJ 2014 Theorem 4.2 conditions applied to the "
           "mapped slice; label inherited from the eSSVI guarantee)",
    proof_note=(
        "Mapped from eSSVI slice theta=0.4, rho=-0.25, psi=1.0 whose GJ "
        "bounds are 0.5 < 4 (both).  to_raw_svi_params gives a=0.1875, "
        "b=0.2, rho=-0.25, m=0.25, sigma=sqrt(0.9375).  Independent literal "
        "g(k) on a dense [-3,3] grid: min g = +0.4037 >= 0 — the mapped "
        "slice is butterfly-free, and the repo's detect_svi must report no "
        "violations."
    ),
)

# ---------------------------------------------------------------------------
# Case 9 — SVI BUTTERFLY-VIOLATING (self-derived raw SVI)
# ---------------------------------------------------------------------------
# A sharp "V" smile: sigma = 0.02 makes the sqrt term nearly |k-m|, giving
# w(k) ~ 0.04 + 0.5*|k| — linear wings with a cusp, whose density is
# negative in a centered band around k = 0.  Verified with the independent
# literal g(k): min g = -0.0641 at k = -0.125, negative k-range
# [-0.2425, 0.2425].  w_min = a + b*sigma*sqrt(1-rho^2) = 0.05 > 0, so the
# only violation is butterfly (no negative-variance flag).
SVI_BUTTERFLY_VIOLATING = GroundTruthCase(
    name="svi_butterfly_violating_vshape",
    model="svi",
    params=SVIParams(a=0.04, b=0.5, rho=0.0, m=0.0, sigma=0.02),
    known_label="butterfly_violation",
    expected_hm_condition=None,
    expected_grid_calendar_free=None,
    source="self-derived (raw SVI density formula, Gatheral 2004 Eq 2.1)",
    proof_note=(
        "w(k) = 0.04 + 0.5*sqrt(k^2 + 0.0004) ~ 0.04 + 0.5*|k|: linear "
        "wings with a cusp.  Independent literal g(k): min g = -0.0641 at "
        "k = -0.125, g < -1e-4 on k in [-0.2425, 0.2425].  w_min = a + "
        "b*sigma*sqrt(1-rho^2) = 0.04 + 0.5*0.02 = 0.05 > 0 so the only "
        "violation is butterfly.  (The published Axel Vogt example, GJ 2014 "
        "Example 3.1, is a second such construction — reserved for the "
        "PAPER_EXAMPLES registry.)"
    ),
)

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

# eSSVI surface cases: every case carries a slice sequence consumed by
# verify_hm_condition / verify_ssvi_calendar_free.
ESSVI_SURFACE_CASES: list[GroundTruthCase] = [
    ESSVI_ARB_FREE,
    ESSVI_VIOLATES_FIRST_ONLY,
    ESSVI_VIOLATES_SECOND_ONLY,
    ESSVI_VIOLATES_BOTH,
    ESSVI_BOUNDARY,
    ESSVI_BOUNDARY_BOTH,
    ESSVI_CALENDAR_VIOLATION,
    ESSVI_CALENDAR_BOUNDARY,
    ESSVI_HM_INSUFFICIENCY,
]

# The boundary cases whose repair-path honesty is asserted (finding 3.10).
BOUNDARY_REPAIR_CASES: list[GroundTruthCase] = [
    ESSVI_BOUNDARY,
    ESSVI_BOUNDARY_BOTH,
]

# The case whose param-level check passes while the grid check fails —
# the non-vacuous 3.10 regression guard (repo's own FIX-6 counterexample).
HM_INSUFFICIENCY_CASE: GroundTruthCase = ESSVI_HM_INSUFFICIENCY

SVI_CASES: list[GroundTruthCase] = [
    SVI_ARB_FREE_FROM_ESSVI,
    SVI_BUTTERFLY_VIOLATING,
]

ALL_CASES: list[GroundTruthCase] = ESSVI_SURFACE_CASES + SVI_CASES


def mapped_raw_svi(case: GroundTruthCase) -> tuple[float, float, float, float, float]:
    """Map an eSSVI slice sequence's LAST slice to raw SVI params.

    Used by the corroboration checks (independent g(k) on the mapped slice).
    """
    last = case.slices[-1]
    return to_raw_svi_params(last.theta, last.rho, last.psi)
