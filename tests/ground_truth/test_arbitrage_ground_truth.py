"""Ground-truth arbitrage tests.

Asserts that the repo's detectors / verifiers AGREE with the hand-derived
labels in ``arbitrage_cases.py``:

- every eSSVI surface case: ``verify_hm_condition`` matches
  ``expected_hm_condition`` and ``verify_ssvi_calendar_free`` matches
  ``expected_grid_calendar_free``;
- every eSSVI case: ``gatheral_jacquier_condition`` matches the hand-computed
  bound arithmetic ``min(4 - theta*psi*(1+|rho|), 4 - theta*psi^2*(1+|rho|))``
  (strict mode for the boundary cases, matching the paper's STRICT condition 1);
- the BOUNDARY cases: the full ``repair(use_ssvi=True)`` path is HONEST —
  ``repair_infeasible`` must exactly equal what the fitted surface's own
  grid/actual-violation checks imply, even when the param-level check
  (``verify_hm_condition``) passes.  This is the test that would FAIL if old
  finding 3.10 (``repair_infeasible`` derived only from ``verify_hm_condition``)
  were reintroduced.  The NON-VACUOUS instance is
  ``ESSVI_HM_INSUFFICIENCY``: its fitted params satisfy ``verify_hm_condition``
  while the native grid check (and ``detect_svi_surface``) report a calendar
  crossing — only the grid fold-in can set the flag honestly;
- the SVI cases: ``detect_svi`` flags exactly the known label, and the
  independent literal ``g(k)`` corroborates it on a dense grid.
"""

from __future__ import annotations

import warnings

import pytest
from pytest import approx

from arbfree_vol.arbitrage.report import ViolationType
from arbfree_vol.arbitrage.svi_detect import detect_svi, detect_svi_surface
from arbfree_vol.repair.engine import repair
from arbfree_vol.ssvi.model import gatheral_jacquier_condition
from arbfree_vol.ssvi.term_structure import (
    verify_hm_condition,
    verify_ssvi_calendar_free,
)

from tests.ground_truth.arbitrage_cases import (
    ALL_CASES,
    BOUNDARY_REPAIR_CASES,
    ESSVI_SURFACE_CASES,
    HM_INSUFFICIENCY_CASE,
    SVI_ARB_FREE_FROM_ESSVI,
    SVI_BUTTERFLY_VIOLATING,
    SVI_CASES,
    dense_g_min,
    gj_hand_residual,
    mapped_raw_svi,
    svi_density_g,
)
from tests.ground_truth.cases import build_essvi_quote_surface

# ---------------------------------------------------------------------------
# eSSVI surface cases: param-level and grid verifiers agree with the labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", ESSVI_SURFACE_CASES, ids=lambda c: c.name
)
def test_essvi_hm_condition_matches_hand_derived(case) -> None:
    """verify_hm_condition must equal the hand-derived HM expectation."""
    got = verify_hm_condition(list(case.slices))
    assert got == case.expected_hm_condition, (
        f"{case.name}: verify_hm_condition={got}, hand-derived "
        f"expected={case.expected_hm_condition}\nproof: {case.proof_note}"
    )


@pytest.mark.parametrize(
    "case", ESSVI_SURFACE_CASES, ids=lambda c: c.name
)
def test_essvi_grid_calendar_check_matches_hand_derived(case) -> None:
    """verify_ssvi_calendar_free must equal the hand-derived grid expectation."""
    got = verify_ssvi_calendar_free(list(case.slices))
    assert got == case.expected_grid_calendar_free, (
        f"{case.name}: verify_ssvi_calendar_free={got}, hand-derived "
        f"expected={case.expected_grid_calendar_free}\nproof: {case.proof_note}"
    )


@pytest.mark.parametrize(
    "case", ESSVI_SURFACE_CASES, ids=lambda c: c.name
)
def test_essvi_gj_diagnostic_matches_hand_computed_bounds(case) -> None:
    """gatheral_jacquier_condition must agree with the literal bound arithmetic.

    Boundary cases use strict=True (the paper's STRICT condition 1: an exact
    equality on the first bound is never certified safe); all other cases use
    the default mode.
    """
    strict = case.name.startswith("essvi_boundary")
    for i, sl in enumerate(case.slices):
        hand = gj_hand_residual(sl.theta, sl.rho, sl.psi, strict=strict)
        got = gatheral_jacquier_condition(sl.theta, sl.rho, sl.psi,
                                          strict=strict)
        assert got == approx(hand, abs=1e-12), (
            f"{case.name} slice {i}: repo residual={got}, hand-derived "
            f"residual={hand} (theta={sl.theta}, rho={sl.rho}, psi={sl.psi})"
        )


def test_arb_free_interior_comfort_pins_harness() -> None:
    """Interior comfort case: everything agrees trivially (harness pin).

    The arb-free interior case is the sanity anchor for the harness — its
    labels must be unambiguous: all GJ bounds with wide margins, HM passes,
    dense-grid calendar passes, and the mapped raw-SVI density comfortably
    positive (min g ~ +0.40).
    """
    case = [c for c in ESSVI_SURFACE_CASES if c.name == "essvi_arb_free_interior"][0]
    assert verify_hm_condition(list(case.slices)) is True
    assert verify_ssvi_calendar_free(list(case.slices)) is True
    a, b, rho, m, sig = mapped_raw_svi(case)
    _, gmin = dense_g_min(a, b, rho, m, sig, n=20001)
    assert gmin > 0.3, f"arb-free interior mapped density min g={gmin}"

    # The repo's param-level diagnostic agrees slice-by-slice.
    for sl in case.slices:
        assert gatheral_jacquier_condition(sl.theta, sl.rho, sl.psi) > 0.0


# ---------------------------------------------------------------------------
# BOUNDARY cases: the full repair path must be HONEST (finding 3.10 guard)
# ---------------------------------------------------------------------------


def _fitted_checks(rep, case):
    """Recompute, on the FITTED surface, the checks the engine should fold
    into repair_infeasible: HM param check, native grid calendar check, and
    the raw-SVI grid detector (the 'actual-violation' check)."""
    fitted = [fs.ssvi for fs in rep.fitted_ssvi_slices]
    hm = verify_hm_condition(fitted) if len(fitted) >= 2 else True
    grid = verify_ssvi_calendar_free(fitted) if len(fitted) >= 2 else True
    svi_slices = [(fs.expiry_time, fs.params) for fs in rep.fitted_slices]
    remaining = detect_svi_surface(svi_slices)
    return fitted, hm, grid, remaining


@pytest.mark.slow
@pytest.mark.parametrize(
    "case", BOUNDARY_REPAIR_CASES, ids=lambda c: c.name
)
def test_boundary_case_repair_is_honest(case) -> None:
    """The FULL repair path must be honest for boundary cases (finding 3.10).

    Runs ``repair(use_ssvi=True)`` on a quote surface built from the case's
    eSSVI slices and asserts ``repair_infeasible`` EXACTLY matches what the
    fitted surface's own grid/actual-violation checks imply.  The key 3.10
    structure: the param-level check (``verify_hm_condition``) passes for
    these calendar-clean cases, so a regression to old finding 3.10
    (``repair_infeasible`` derived only from ``verify_hm_condition``) would
    report False here while the grid checks may fail — the assertion would
    break.

    Note on the construction: the hard eSSVI optimizer can relocate a
    boundary slice to a feasible smile pinned on the GJ condition-2 boundary
    (a poor-fit but constraint-satisfying smile whose RMSE is reported), in
    which case the fitted surface's own grid checks pass and
    ``repair_infeasible=False`` IS the honest outcome.  The non-vacuous
    instance of the guard is ``ESSVI_HM_INSUFFICIENCY`` (separate test), whose
    fitted params pass ``verify_hm_condition`` while the grid check fails.
    """
    from arbfree_vol.ssvi.term_structure import verify_hm_condition as _hm

    surface = build_essvi_quote_surface(
        [(0.25, case.slices[0]), (0.5, case.slices[1]),
         (2.0, case.slices[-1])],
        n_k=13, k_lo=-0.6, k_hi=0.6,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy delta_grad chatter on escapes
        rep = repair(surface, use_ssvi=True)

    fitted, hm, grid, remaining = _fitted_checks(rep, case)

    # The param-level check passes for these calendar-clean cases — this is
    # exactly the 3.10 trap: only the grid/actual-violation fold-in can flag.
    assert hm is True, (
        f"{case.name}: fitted params violate HM ({hm}); the boundary case is "
        f"calendar-clean by construction"
    )

    expected_infeasible = (not hm) or (not grid) or (not remaining.is_arbitrage_free)
    assert rep.repair_infeasible == expected_infeasible, (
        f"{case.name}: repair_infeasible={rep.repair_infeasible} but the "
        f"fitted surface's own checks imply {expected_infeasible} "
        f"(hm={hm}, grid={grid}, remaining_violations="
        f"{[(str(v.kind), round(v.magnitude, 6)) for v in remaining.violations]}). "
        f"If hm passed while a grid check failed, only the grid fold-in can "
        f"set the flag — a 3.10 regression would report False."
    )


@pytest.mark.slow
def test_hm_insufficiency_repair_is_honest_and_non_vacuous() -> None:
    """The 3.10 guard fires on the H&M-sufficiency counterexample.

    Builds a quote surface from the repo's documented FIX-6 counterexample
    pair (docs/review_campaign.md, Item-2): ``verify_hm_condition`` PASSES on
    the pair while the slices cross (w2 - w1 = -0.00153 at k = 0.68).  After
    ``repair(use_ssvi=True)``:

    - ``verify_hm_condition`` on the FITTED params is True (param-level
      passes) — the old finding 3.10 code would therefore report
      ``repair_infeasible=False``;
    - the native grid calendar check on the FITTED params is False and
      ``detect_svi_surface`` reports a CALENDAR violation — the load-bearing
      actual-violation checks;
    - ``repair_infeasible`` must be True.

    If 3.10 were reintroduced this test FAILS (the flag would be False).
    """
    case = HM_INSUFFICIENCY_CASE
    surface = build_essvi_quote_surface(
        [(0.25, case.slices[0]), (1.0, case.slices[1])],
        n_k=13, k_lo=-0.6, k_hi=0.6,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = repair(surface, use_ssvi=True)

    fitted, hm, grid, remaining = _fitted_checks(rep, case)

    assert len(fitted) == 2, f"expected both slices fitted, got {len(fitted)}"
    assert rep.metrics.n_rejected == 0, (
        "the counterexample slices are butterfly-safe, so no quotes may be "
        "rejected before the eSSVI fit"
    )

    # (a) param-level passes
    assert hm is True, (
        f"fitted HM check must pass (the counterexample satisfies HM): {hm}"
    )
    # (b) the actual-violation checks fail
    assert grid is False, (
        "fitted native-SSVI grid calendar check must fail: the slices cross "
        "(w2 - w1 = -0.00153 at k = 0.68)"
    )
    cal_kinds = [v.kind for v in remaining.violations]
    assert ViolationType.CALENDAR in cal_kinds, (
        f"detect_svi_surface must report a calendar violation, got "
        f"{[(str(v.kind), round(v.magnitude, 6)) for v in remaining.violations]}"
    )
    # (c) the honest flag
    assert rep.repair_infeasible is True, (
        "repair_infeasible must be True: verify_hm_condition passes but the "
        "grid/actual-violation checks fail — a 3.10-style flag derived only "
        "from the param-level check would be False here"
    )


# ---------------------------------------------------------------------------
# SVI cases: detector flags exactly the known label + independent g(k)
# ---------------------------------------------------------------------------


def test_svi_arb_free_from_essvi_matches_label() -> None:
    """detect_svi reports no violations; independent g(k) >= 0 on the grid."""
    case = SVI_ARB_FREE_FROM_ESSVI
    p = case.params

    report = detect_svi(p)
    assert report.is_arbitrage_free, (
        f"mapped arb-free slice must pass detect_svi, got "
        f"{[(str(v.kind), round(v.magnitude, 6)) for v in report.violations]}"
    )

    _, gmin = dense_g_min(p.a, p.b, p.rho, p.m, p.sigma, n=20001)
    assert gmin >= 0.0, (
        f"independent literal g(k) must be non-negative, min g={gmin}"
    )
    # The mapped slice is comfortably interior (min g ~ +0.40).
    assert gmin > 0.2, f"expected a comfortable interior margin, min g={gmin}"


def test_svi_butterfly_violating_matches_label() -> None:
    """detect_svi flags exactly BUTTERFLY; independent g(k) < 0 in the band."""
    case = SVI_BUTTERFLY_VIOLATING
    p = case.params

    report = detect_svi(p)
    kinds = [v.kind for v in report.violations]
    assert len(kinds) == 1 and kinds[0] == "butterfly", (
        f"expected exactly one BUTTERFLY violation, got "
        f"{[(str(v.kind), round(v.magnitude, 6)) for v in report.violations]}"
    )

    # Independent literal g(k) corroboration: negative in the documented band.
    grid = [k / 1000.0 for k in range(-3000, 3001)]
    neg = [k for k in grid if svi_density_g(k, p.a, p.b, p.rho, p.m, p.sigma) < -1e-4]
    assert neg, "independent g(k) must dip below -1e-4 somewhere on [-3,3]"
    assert min(neg) <= -0.2 and max(neg) >= 0.2, (
        f"expected the negative band to cover [-0.2, 0.2], got "
        f"[{min(neg):.3f}, {max(neg):.3f}]"
    )
    _, gmin = dense_g_min(p.a, p.b, p.rho, p.m, p.sigma, n=20001)
    assert gmin < -0.05, f"expected a clear negative density, min g={gmin}"


# ---------------------------------------------------------------------------
# Registry sanity (anti-circularity)
# ---------------------------------------------------------------------------

def test_all_cases_carry_hand_derived_labels() -> None:
    """Every case must carry a source and a proof note (anti-circularity)."""
    for case in ALL_CASES:
        assert case.source.startswith("self-derived"), (
            f"{case.name}: source must be self-derived, got {case.source!r}"
        )
        assert case.proof_note, f"{case.name}: proof_note must be non-empty"
        assert case.name, "every case needs a name"


def test_svi_case_params_are_the_mapped_literals() -> None:
    """The SVI case params are the analytic literals, not repo fits.

    For the arb-free-from-eSSVI case, a = (theta/2)(1-rho^2) = 0.1875,
    b = theta*psi/2 = 0.2, m = -rho/psi = 0.25,
    sigma = sqrt(1-rho^2)/psi = sqrt(0.9375) — computed by hand here.
    """
    from math import sqrt

    p = SVI_ARB_FREE_FROM_ESSVI.params
    assert p.a == approx(0.1875, abs=1e-12)
    assert p.b == approx(0.2, abs=1e-12)
    assert p.m == approx(0.25, abs=1e-12)
    assert p.sigma == approx(sqrt(0.9375), abs=1e-12)
    assert p.rho == approx(-0.25, abs=1e-12)
