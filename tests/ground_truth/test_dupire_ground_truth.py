"""Ground-truth Dupire local-volatility tests.

Two closed-form cases:

1. CONSTANT-VOL: w(k, T) = sigma^2 * T  ->  local vol = sigma everywhere.
   The repo's ``dupire`` on a dense interior grid must recover sigma within
   a documented FD tolerance.  For this surface the FD is actually exact
   (dw/dk = d2w/dk2 = 0 and w linear in T), so the tolerance is tight
   (rel 1e-9; measured deviation ~1e-14).

2. LINEAR-IN-K: w(k, T) = sigma^2 * T * (1 + beta*k) with the closed-form
   local vol derived in ``dupire_cases.py`` (Gatheral 2004 Eq 1.10) and
   implemented LITERALLY there, independent of the repo's ``local_vol.py``.

STOPPED-ON FINDING — RESOLVED (2026-08-13)
-----------------------------------------
The repo's ``dupire`` previously did NOT agree with the hand-derived closed
form for the linear-in-k case: the measured relative deviation was
4e-3..8e-3, well above the documented FD tolerance of 2e-3.  Investigation
found a genuine repo bug in ``pricing/local_vol.py::_d2w_dk2``: it applied
the symmetric second-difference formula ``(w+ - 2w0 + w-)/dk^2`` with
ASYMMETRIC k-steps (equal K-steps give unequal k-steps because
k = ln(K/F)), which injected a spurious ``-w'(k)`` term into the second
derivative.  Measured on the linear branch: true d2w = 0, repo d2w = -b
(the smile slope).  A SECOND same-class bug was found in ``_dw_dT``, which
took the T-derivative at fixed absolute strike K instead of fixed
log-moneyness k, injecting a spurious ``(r-q)*dw/dk`` term (first-order in
the smile slope for non-zero carry).  Both are fixed with the non-uniform
central second-difference stencil and the fixed-k T-stencil; the xfail
below has been flipped to a passing test.  The repo-vs-closed-form
deviation is now ~3.4e-7 relative (tolerance 2e-3).  The closed form
itself is validated by a non-xfail test against an independent
price-space Dupire finite-difference (which uses no total-variance-space
formula at all).
"""

from __future__ import annotations

import math

from pytest import approx

from arbfree_vol.pricing.local_vol import dupire
from arbfree_vol.surface.interpolate import total_variance_at

from tests.ground_truth.dupire_cases import (
    CONSTANT_VOL_INTERIOR_ROWS,
    CONSTANT_VOL_MATURITIES,
    CONSTANT_VOL_STRIKES,
    FD_REL_TOL_CONST,
    FD_REL_TOL_LINEAR,
    LINEAR_IN_K_INTERIOR_ROWS,
    LINEAR_IN_K_MATURITIES,
    LINEAR_IN_K_STRIKES,
    NONFLAT_REFERENCE_POINTS,
    SIGMA_CONST,
    SPOT,
    build_constant_vol_surface,
    build_linear_in_k_surface,
    build_nonflat_svi_surface,
    closed_form_linear_sigma_loc,
    closed_form_nonflat_svi_sigma_loc,
    price_space_dupire_sigma_loc,
)

# ---------------------------------------------------------------------------
# Case 1 — constant-vol BS surface
# ---------------------------------------------------------------------------


def test_constant_vol_surface_recovers_sigma() -> None:
    """dupire on the flat surface recovers sigma on the interior grid.

    Interior maturity rows only (strictly inside [0.5, 2.0], away from the
    FD stencil edges).  The flat smile has zero k-derivatives and w linear
    in T, so the repo's finite differences are exact: the documented
    tolerance rel 1e-9 sits ~5 orders above the measured ~1e-14 deviation
    and ~7 orders below any real local-vol structure.
    """
    fs = build_constant_vol_surface()
    lv = dupire(fs, list(CONSTANT_VOL_STRIKES), list(CONSTANT_VOL_MATURITIES))

    for iT in CONSTANT_VOL_INTERIOR_ROWS:
        for iK, K in enumerate(CONSTANT_VOL_STRIKES):
            val = lv.grid[iT][iK]
            assert val == approx(SIGMA_CONST, rel=FD_REL_TOL_CONST), (
                f"K={K:.2f}, T={CONSTANT_VOL_MATURITIES[iT]}: got {val:.10f}, "
                f"expected {SIGMA_CONST}"
            )


def test_constant_vol_analytic_label_statement() -> None:
    """Hand-derived label for the constant-vol case (documented in code).

    w(k, T) = sigma^2 * T: dw/dk = 0, d2w/dk2 = 0, dw/dT = sigma^2, so the
    Gatheral denominator is 1 and sigma_loc^2 = sigma^2 exactly — local vol
    is sigma at every interior (k, T).
    """
    for k in (-0.2, 0.0, 0.2):
        for T in (0.75, 1.0, 1.5):
            w = total_variance_at(build_constant_vol_surface(),
                                  SPOT * math.exp(k), T)
            assert w == approx(SIGMA_CONST ** 2 * T, rel=1e-12)


# ---------------------------------------------------------------------------
# Case 2 — linear-in-k surface
# ---------------------------------------------------------------------------


def test_linear_in_k_surface_matches_linear_form_on_grid() -> None:
    """The repo-consumable surface really is w = sigma^2 T (1 + beta k).

    The SVI branch construction (kink at m=-1, smoothing 1e-3) reproduces
    the linear form to ~1e-8 in total variance — two orders of magnitude
    below the FD tolerance used below.
    """
    from tests.ground_truth.dupire_cases import BETA_LIN, SIGMA_LIN

    fs = build_linear_in_k_surface()
    for T in (0.75, 1.0, 1.25, 1.5):
        for k in (-0.5, -0.2, 0.0, 0.3, 0.5):
            K = SPOT * math.exp(k)
            got = total_variance_at(fs, K, T)
            want = SIGMA_LIN ** 2 * T * (1.0 + BETA_LIN * k)
            assert got == approx(want, abs=1e-6), (
                f"T={T}, k={k}: surface w={got}, linear w={want}"
            )


def test_linear_in_k_closed_form_matches_price_space_fd() -> None:
    """The closed form is the correct label (independent ground truth).

    Compares the literal closed form against an independent price-space
    Dupire finite-difference computed on the repo's OWN surface (Black-
    Scholes prices -> [dC/dT]/[(1/2)K^2 C_KK]).  The two agree within
    ~3.2e-4 relative — the finite-difference error of the price-space grid
    (first/second-order FD over dK = 1% of K and dT = 1e-4), so the
    documented 5e-4 tolerance is several times the observed FD error —
    proving the hand-derived label, not the repo's total-variance-space
    output, is the true local vol of the surface.
    """
    fs = build_linear_in_k_surface()
    for T in (0.75, 1.0, 1.25, 1.5):
        for k in (-0.4, -0.2, 0.0, 0.2, 0.4):
            K = SPOT * math.exp(k)
            closed = closed_form_linear_sigma_loc(k, T)
            price_fd = price_space_dupire_sigma_loc(fs, K, T)
            assert price_fd == approx(closed, rel=5e-4), (
                f"k={k}, T={T}: price-FD={price_fd:.8f} vs closed "
                f"form={closed:.8f}"
            )


def test_linear_in_k_repo_dupire_matches_closed_form() -> None:
    """``dupire`` must reproduce the closed-form local vol on the interior grid.

    Interior maturity rows only.  This test was the module's honest
    ``xfail(strict=True)`` pin while ``pricing/local_vol.py::_d2w_dk2``
    applied a symmetric second-difference formula to ASYMMETRIC k-steps
    (equal K-steps give unequal k-steps since k = ln(K/F)), injecting a
    spurious -w'(k) term (measured d2w = -b on a linear branch where the
    true d2w = 0).  The fix (non-uniform central stencil + fixed-k ∂w/∂T)
    removed the bias; the measured deviation on this grid is now ~3.4e-7
    relative, well inside the documented FD tolerance of 2e-3.
    """
    fs = build_linear_in_k_surface()
    lv = dupire(fs, list(LINEAR_IN_K_STRIKES), list(LINEAR_IN_K_MATURITIES))

    for iT in LINEAR_IN_K_INTERIOR_ROWS:
        for iK, K in enumerate(LINEAR_IN_K_STRIKES):
            k = math.log(K / SPOT)
            expected = closed_form_linear_sigma_loc(k, LINEAR_IN_K_MATURITIES[iT])
            got = lv.grid[iT][iK]
            assert got == approx(expected, rel=FD_REL_TOL_LINEAR), (
                f"K={K:.3f} (k={k:.2f}), T={LINEAR_IN_K_MATURITIES[iT]}: "
                f"repo={got:.8f}, closed form={expected:.8f}"
            )


# ---------------------------------------------------------------------------
# Case 3 — non-flat SVI surface (the repo's test_local_vol.py pin)
# ---------------------------------------------------------------------------


def test_nonflat_surface_total_variance_matches_svi_interpolation() -> None:
    """The repo-consumable non-flat surface is the SVI interpolation.

    Pins the exact surface semantics the closed-form evaluator replicates:
    each slice's total variance evaluated at ITS OWN forward's
    log-moneyness (``k_i = ln(K/F_i)``), linearly interpolated in T.  This
    is the executable statement of the "per-slice forwards differ when
    r != q" behaviour that makes the surface NOT ``w = T*f(k)``.
    """
    from arbfree_vol.surface.interpolate import _forward_at

    from tests.ground_truth.dupire_cases import _svi_literal_w

    fs = build_nonflat_svi_surface()
    sl_low, sl_high = fs.fitted_slices[0], fs.fitted_slices[1]
    p_low, p_high = sl_low.params, sl_high.params
    dT = sl_high.expiry_time - sl_low.expiry_time
    F_low = _forward_at(fs, sl_low.expiry_time)
    F_high = _forward_at(fs, sl_high.expiry_time)
    for K, T in NONFLAT_REFERENCE_POINTS:
        theta = (T - sl_low.expiry_time) / dT
        wA, _, _ = _svi_literal_w(math.log(K / F_low), p_low.a, p_low.b,
                                  p_low.rho, p_low.m, p_low.sigma)
        wB, _, _ = _svi_literal_w(math.log(K / F_high), p_high.a, p_high.b,
                                  p_high.rho, p_high.m, p_high.sigma)
        want = (1.0 - theta) * wA + theta * wB
        got = total_variance_at(fs, K, T)
        assert got == approx(want, rel=1e-12), (
            f"K={K}, T={T}: surface w={got}, literal interpolation={want}"
        )


def test_nonflat_closed_form_matches_price_space_fd() -> None:
    """The non-flat SVI closed form is the correct label (independent GT).

    Compares the literal closed form against the model-independent
    price-space Dupire finite difference computed on the repo's OWN surface
    (Black-Scholes prices on the repo's ACTUAL linear forward curve ->
    [dC/dT + mu*K dC/dK + (r - mu) C]/(0.5 K^2 C_KK) with the CORRECTED
    forward drift mu = F'(T)/F(T), per the FIX-GT work).  The two agree
    within ~1.8e-5 relative on the interior reference points (the
    price-space FD grid error for dK = 0.2% of K, dT = 1e-4); the
    documented 1e-4 tolerance is several times the observed FD error —
    proving the hand-derived closed form, not the repo's total-variance-
    space output, is the true local vol of the surface.  This is the
    executable provenance behind the regression test in
    ``tests/test_local_vol.py``.
    """
    fs = build_nonflat_svi_surface()
    for K, T in NONFLAT_REFERENCE_POINTS:
        closed = closed_form_nonflat_svi_sigma_loc(fs, K, T)
        price_fd = price_space_dupire_sigma_loc(fs, K, T, dK_rel=2e-3, dT=1e-4)
        assert price_fd == approx(closed, rel=1e-4), (
            f"K={K}, T={T}: price-FD={price_fd:.8f} vs closed "
            f"form={closed:.8f}"
        )
