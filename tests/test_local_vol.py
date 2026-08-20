"""Tests for Dupire local volatility (pricing/local_vol.py).

All tests are synthetic — no network, no yfinance.
"""

import math

import pytest
from pytest import approx

from arbfree_vol.svi.model import SVIParams
from arbfree_vol.models.fitted import FittedSlice, FittedSurface
from arbfree_vol.pricing.local_vol import (
    dupire_at,
    dupire,
    _d2w_dk2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _forward(T: float, spot: float = 100.0, r: float = 0.05,
             q: float = 0.0) -> float:
    """Forward price for a given expiry time."""
    return spot * math.exp((r - q) * T)


def _flat_fitted_surface(
    T_low: float = 0.5,
    T_high: float = 2.0,
    sigma: float = 0.2,
    spot: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
) -> FittedSurface:
    """Build a two-slice fitted surface with a flat (b=0) smile.

    Each slice has ``a = sigma² × T``, ``b = 0``, so total variance is
    constant at ``sigma² × T`` regardless of log-moneyness.  This makes
    ``iv_at(K, T) = sigma`` for any strike *K*.
    """
    sl_low = FittedSlice(
        expiry_time=T_low,
        params=SVIParams(
            a=sigma ** 2 * T_low, b=0.0, rho=0.0, m=0.0, sigma=0.2
        ),
        rmse=0.0,
        forward_price=_forward(T_low, spot, r, q),
        n_quotes_total=5,
        n_quotes_used=5,
    )
    sl_high = FittedSlice(
        expiry_time=T_high,
        params=SVIParams(
            a=sigma ** 2 * T_high, b=0.0, rho=0.0, m=0.0, sigma=0.2
        ),
        rmse=0.0,
        forward_price=_forward(T_high, spot, r, q),
        n_quotes_total=5,
        n_quotes_used=5,
    )

    return FittedSurface(
        spot=spot,
        risk_free=r,
        div_yield=q,
        forward_curve=(
            (T_low, _forward(T_low, spot, r, q)),
            (T_high, _forward(T_high, spot, r, q)),
        ),
        fitted_slices=(sl_low, sl_high),
    )


# ---------------------------------------------------------------------------
# Test: flat surface → flat local volatility
# ---------------------------------------------------------------------------
class TestDupireFlatSurface:
    """For a flat smile (b=0) the Dupire formula should recover sigma."""

    def test_dupire_flat_returns_flat(self) -> None:
        """dupire_at returns sigma (within FD tolerance) on interior grid."""
        sigma = 0.2
        fs = _flat_fitted_surface(0.5, 2.0, sigma)

        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.5, 1.0, 1.5, 2.0]

        lv = dupire(fs, strikes, maturities)

        # Interior maturity rows: indices 1 and 2 (T=1.0, T=1.5)
        # Boundary maturities (0.5 and 2.0) may have jitter from FD.
        for iT in (1, 2):
            for iK in range(len(strikes)):
                val = lv.grid[iT][iK]
                assert val == approx(sigma, rel=5e-3), (
                    f"Maturity={maturities[iT]}, strike={strikes[iK]}: "
                    f"got {val:.6f}, expected {sigma}"
                )

    def test_dupire_grid_shape(self) -> None:
        """Grid dimensions match the input arrays."""
        sigma = 0.2
        fs = _flat_fitted_surface(0.5, 2.0, sigma)

        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.5, 1.0, 1.5, 2.0]

        lv = dupire(fs, strikes, maturities)
        assert len(lv.grid) == 4, f"Expected 4 maturity rows, got {len(lv.grid)}"
        for row in lv.grid:
            assert len(row) == 5, (
                f"Expected 5 strikes per row, got {len(row)}"
            )

    def test_dupire_at_exact_atm(self) -> None:
        """dupire_at at ATM (K=spot) on interior T returns sigma."""
        sigma = 0.2
        fs = _flat_fitted_surface(0.5, 2.0, sigma)

        # T=1.0 (interior)
        val = dupire_at(fs, K=100.0, T=1.0)
        assert val == approx(sigma, rel=5e-3), (
            f"dupire_at(ATM, T=1.0) = {val:.6f}, expected {sigma}"
        )


# ---------------------------------------------------------------------------
# Test: calendar arb raises
# ---------------------------------------------------------------------------
class TestDupireCalendarArb:
    """dw/dT < 0 should raise ValueError."""

    def test_dupire_calendar_arb_raises(self) -> None:
        """Earlier slice has larger total variance than later slice."""
        spot = 100.0
        r = 0.05
        q = 0.0
        T_low = 0.5
        T_high = 2.0

        # Later slice has LOWER total variance → w decreasing with T
        sl_low = FittedSlice(
            expiry_time=T_low,
            params=SVIParams(
                a=0.3 ** 2 * T_low, b=0.0, rho=0.0, m=0.0, sigma=0.2
            ),
            rmse=0.0,
            forward_price=_forward(T_low, spot, r, q),
            n_quotes_total=5,
            n_quotes_used=5,
        )
        sl_high = FittedSlice(
            expiry_time=T_high,
            params=SVIParams(
                a=0.1 ** 2 * T_high, b=0.0, rho=0.0, m=0.0, sigma=0.2
            ),
            rmse=0.0,
            forward_price=_forward(T_high, spot, r, q),
            n_quotes_total=5,
            n_quotes_used=5,
        )

        fs = FittedSurface(
            spot=spot,
            risk_free=r,
            div_yield=q,
            forward_curve=(
                (T_low, _forward(T_low, spot, r, q)),
                (T_high, _forward(T_high, spot, r, q)),
            ),
            fitted_slices=(sl_low, sl_high),
        )

        # At interior T where dw/dT < 0 dupire_at should raise.
        with pytest.raises(ValueError, match="Calendar arbitrage"):
            dupire_at(fs, K=spot, T=1.5)


# ---------------------------------------------------------------------------
# Test: interior local vol positive for a normal SVI smile
# ---------------------------------------------------------------------------
class TestDupireNormalSmile:
    """Non-trivial SVI smile with non-decreasing total variance in T."""

    def test_dupire_interior_positive_for_normal_smile(self) -> None:
        """Local vol is positive and not nan for all interior cells."""
        spot = 100.0
        r = 0.05
        q = 0.0
        T_low = 0.5
        T_high = 2.0

        # Reference SVI parameters at T=1.0 (Gatheral-ish values)
        a_ref = 0.04
        b_ref = 0.4
        rho_ref = -0.4
        m_ref = 0.05
        sigma_ref = 0.15

        # Scale a, b linearly with T so that w(k,T) = T * f(k) for some f,
        # keeping rho, m, sigma_param identical.
        sl_low = FittedSlice(
            expiry_time=T_low,
            params=SVIParams(
                a=a_ref * T_low,
                b=b_ref * T_low,
                rho=rho_ref,
                m=m_ref,
                sigma=sigma_ref,
            ),
            rmse=0.0,
            forward_price=_forward(T_low, spot, r, q),
            n_quotes_total=5,
            n_quotes_used=5,
        )
        sl_high = FittedSlice(
            expiry_time=T_high,
            params=SVIParams(
                a=a_ref * T_high,
                b=b_ref * T_high,
                rho=rho_ref,
                m=m_ref,
                sigma=sigma_ref,
            ),
            rmse=0.0,
            forward_price=_forward(T_high, spot, r, q),
            n_quotes_total=5,
            n_quotes_used=5,
        )

        fs = FittedSurface(
            spot=spot,
            risk_free=r,
            div_yield=q,
            forward_curve=(
                (T_low, _forward(T_low, spot, r, q)),
                (T_high, _forward(T_high, spot, r, q)),
            ),
            fitted_slices=(sl_low, sl_high),
        )

        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.5, 1.0, 1.5, 2.0]

        lv = dupire(fs, strikes, maturities)

        # Interior maturities: indices 1 and 2 (T=1.0, T=1.5)
        for iT in (1, 2):
            for iK in range(len(strikes)):
                val = lv.grid[iT][iK]
                assert not math.isnan(val), (
                    f"nan at T={maturities[iT]}, K={strikes[iK]}"
                )
                assert val > 0.0, (
                    f"Non-positive local vol {val:.6f} at "
                    f"T={maturities[iT]}, K={strikes[iK]}"
                )


# ---------------------------------------------------------------------------
# Test: out-of-surface raises
# ---------------------------------------------------------------------------
class TestDupireOutOfSurface:
    """T outside the fitted surface range raises ValueError."""

    def test_dupire_out_of_surface_raises(self) -> None:
        """T below earliest slice expiry raises ValueError."""
        sigma = 0.2
        fs = _flat_fitted_surface(0.5, 2.0, sigma)

        with pytest.raises(ValueError):
            dupire_at(fs, K=100.0, T=0.1)


# ---------------------------------------------------------------------------
# Test: regression — non-flat smile exact values
# ---------------------------------------------------------------------------
class TestDupireNonFlatSmileExact:
    """Regression test: non-flat SVI smile must produce known local vols."""

    def test_dupire_non_flat_exact_values(self) -> None:
        """dupire_at on the non-flat SVI smile matches an INDEPENDENT
        closed-form reference within the documented FD tolerance.

        The expected values are computed at test time from the independent
        closed-form evaluator ``closed_form_nonflat_svi_sigma_loc`` in
        ``tests/ground_truth/dupire_cases.py`` — Gatheral (2004) Eq 1.10
        with the surface's analytic SVI derivatives and the exact fixed-k
        ∂w/∂T (the FIX-GT corrected convention), never the repo's
        ``local_vol.py``.  For regression visibility the previously pinned
        opaque literals, reproduced exactly by the evaluator, were:
            (90,1.0)=0.5966932352, (90,1.5)=0.6942597587,
            (100,1.0)=0.3422958235, (100,1.5)=0.3442086699,
            (110,1.0)=0.2225186276, (110,1.5)=0.1983872723
        The evaluator's OWN provenance (agreement with an independent
        price-space Dupire FD using the corrected forward drift
        mu = F'(T)/F(T)) is asserted in test_dupire_ground_truth.py.
        The repo's FD stencil (dK = K*1e-3, dT = 1e-3) adds ~2-3.5e-6
        relative, so the honest tolerance below is 1e-4 (documented FD
        stencil limits; see FD_REL_TOL_NONFLAT in dupire_cases.py) — orders
        of magnitude tighter than the ~12% bias the OLD symmetric-stencil
        values certified.
        """
        from tests.ground_truth.dupire_cases import (
            NONFLAT_REFERENCE_POINTS,
            build_nonflat_svi_surface,
            closed_form_nonflat_svi_sigma_loc,
        )

        fs = build_nonflat_svi_surface()

        for K, T in NONFLAT_REFERENCE_POINTS:
            ref = closed_form_nonflat_svi_sigma_loc(fs, K, T)
            lv = dupire_at(fs, K, T)
            assert lv == approx(ref, rel=1e-4), (
                f"K={K}, T={T}: got {lv:.10f}, expected {ref:.10f}"
            )


# ---------------------------------------------------------------------------
# Test: regression — non-uniform k-grid second-difference
# ---------------------------------------------------------------------------
class TestDupireNonUniformSecondDifference:
    """The second-derivative stencil must handle the non-uniform k-grid.

    The repo's stencil steps in *absolute strike* (``K ± dK``), so the
    implied k-grid (``k = ln(K/F)``) is NOT uniform.  The old symmetric
    second-difference ``(w⁺ − 2w⁰ + w⁻)/dk²`` injected a spurious
    ``-w'(k)`` term on that grid.  These tests pin the corrected
    non-uniform stencil.
    """

    def test_d2w_dk2_zero_on_linear_in_k_branch(self) -> None:
        """d²w/dk² ≈ 0 on a linear-in-k total-variance slice.

        On a slice whose total variance is ``w = sigma²·T·(1 + beta·k)``
        the true second derivative is zero.  The pre-fix symmetric formula
        returned ``-b = -sigma²·T·beta`` (the smile slope); the corrected
        non-uniform stencil must return ~0 instead.
        """
        from tests.ground_truth.dupire_cases import (
            BETA_LIN,
            SIGMA_LIN,
            build_linear_in_k_surface,
        )

        fs = build_linear_in_k_surface()
        F_T = 100.0  # SPOT; r = q = 0 in the ground-truth cases module

        # True second derivative on the linear branch: exactly 0.
        # Pre-fix symmetric formula would return -b = -SIGMA_LIN²·T·BETA_LIN
        # (e.g. -0.028125 at T=1.5) — two orders of magnitude beyond the
        # tolerance here.
        for T in (0.75, 1.0, 1.25, 1.5):
            for k in (-0.4, -0.2, 0.0, 0.2, 0.4):
                K = 100.0 * math.exp(k)
                d2 = _d2w_dk2(fs, K, T, F_T)
                assert d2 == approx(0.0, abs=1e-4), (
                    f"T={T}, k={k}: d2w/dk2={d2:.8f}, expected ~0 "
                    f"(pre-fix symmetric stencil gave "
                    f"{-SIGMA_LIN ** 2 * T * BETA_LIN:.8f})"
                )

    def test_d2w_dk2_reduces_to_symmetric_on_uniform_grid(self) -> None:
        """The non-uniform stencil reduces to the symmetric formula when
        the k-grid IS uniform (equal k-steps)."""
        # Build a 2-slice surface (any smile works); the point here is the
        # algebra, verified directly: on a uniform k-grid
        # h⁺ = h⁻ = h the stencil is
        #   2/(h⁺+h⁻)·[(w⁺−w⁰)/h⁺ − (w⁰−w⁻)/h⁻]
        # = 1/h·[(w⁺−w⁰)/h − (w⁰−w⁻)/h]
        # = (w⁺ − 2w⁰ + w⁻)/h².
        # Numerically simulate the formula with equal k-steps and confirm it
        # equals the symmetric form.
        w0, wm, wp, h = 0.4, 0.3, 0.55, 0.05
        symmetric = (wp - 2.0 * w0 + wm) / (h * h)
        non_uniform = 2.0 / (h + h) * ((wp - w0) / h - (w0 - wm) / h)
        assert non_uniform == approx(symmetric, rel=1e-12)


# ---------------------------------------------------------------------------
# Test: _dw_dk forward-difference branch and nan guards
# ---------------------------------------------------------------------------
class TestFirstDerivativeBranches:
    """Exercise the low-strike forward difference and nan-guard branches of
    the first derivative of total variance w.r.t. log-moneyness."""

    def test_dw_dk_forward_diff_at_low_strike(self) -> None:
        """K ≤ dK selects the forward-difference branch (no left point
        exists).  For a flat smile the slope is ~0 and the branch must
        return a finite value, not nan."""
        from arbfree_vol.pricing.local_vol import _dw_dk

        fs = _flat_fitted_surface(0.5, 2.0, 0.2)
        F_T = _forward(1.0)

        # K=1e-4, dK defaults to 1e-3 → K - dK <= 0 → forward branch
        dwdk = _dw_dk(fs, K=1e-4, T=1.0, F_T=F_T)
        assert not math.isnan(dwdk)
        assert dwdk == approx(0.0, abs=1e-9)

    def test_dw_dk_central_diff(self) -> None:
        """Interior K uses the central difference and matches the forward
        branch value on a flat smile."""
        from arbfree_vol.pricing.local_vol import _dw_dk

        fs = _flat_fitted_surface(0.5, 2.0, 0.2)
        F_T = _forward(1.0)

        dwdk = _dw_dk(fs, K=100.0, T=1.0, F_T=F_T)
        assert not math.isnan(dwdk)
        assert dwdk == approx(0.0, abs=1e-9)

    def test_dw_dk_central_precision_nan_guard(self) -> None:
        """When the central-difference k-step collapses below 1e-15 the
        branch returns nan."""
        from arbfree_vol.pricing.local_vol import _dw_dk

        fs = _flat_fitted_surface(0.5, 2.0, 0.2)

        # Central branch: K > dK, dk = 0.5*log((K+dK)/(K-dK)).  With a tiny
        # dK relative to K, (K+dK)/(K-dK) ~ 1 -> dk collapses.
        val = _dw_dk(fs, K=100.0, T=1.0, F_T=_forward(1.0), dK=1e-18)
        assert math.isnan(val)


class TestSecondDerivativeGuards:
    """The second-derivative stencil guards: edge (no left point) and
    precision (degenerate k-step) both return nan."""

    def test_d2w_dk2_edge_guard_returns_nan(self) -> None:
        """K - dK <= 0 (no left point for the central second difference)
        returns nan rather than crashing or extrapolating."""
        from arbfree_vol.pricing.local_vol import _d2w_dk2

        fs = _flat_fitted_surface(0.5, 2.0, 0.2)
        F_T = _forward(1.0)

        val = _d2w_dk2(fs, K=1e-4, T=1.0, F_T=F_T)
        assert math.isnan(val)

    def test_d2w_dk2_normal_stencil_finite(self) -> None:
        """Interior K on a flat smile gives a finite (zero) second
        derivative."""
        from arbfree_vol.pricing.local_vol import _d2w_dk2

        fs = _flat_fitted_surface(0.5, 2.0, 0.2)
        F_T = _forward(1.0)

        val = _d2w_dk2(fs, K=100.0, T=1.0, F_T=F_T)
        assert not math.isnan(val)
        assert val == approx(0.0, abs=1e-9)

    def test_d2w_dk2_precision_nan_guard(self) -> None:
        """When either k-space half-step collapses below 1e-15 the stencil
        returns nan (degenerate grid, second derivative undefined)."""
        from arbfree_vol.pricing.local_vol import _d2w_dk2

        fs = _flat_fitted_surface(0.5, 2.0, 0.2)

        # h_plus = log((K+dK)/K), h_minus = log(K/(K-dK)); a tiny dK
        # collapses both half-steps.
        val = _d2w_dk2(fs, K=100.0, T=1.0, F_T=_forward(1.0), dK=1e-18)
        assert math.isnan(val)


# ---------------------------------------------------------------------------
# Test: dupire_at nan propagation via denominator
# ---------------------------------------------------------------------------
class TestDupireDenominatorNan:
    """dupire_at maps a non-positive Dupire denominator to nan (local
    volatility undefined there), rather than raising or returning a
    negative square root."""

    def test_dupire_at_nan_at_extreme_wing(self) -> None:
        """A steep smile (large b, rho near -1) drives the Dupire
        denominator negative at extreme moneyness → dupire_at returns nan."""
        from arbfree_vol.svi.model import SVIParams
        from arbfree_vol.models.fitted import FittedSlice, FittedSurface

        spot = 100.0
        r = 0.05
        q = 0.0

        sl_low = FittedSlice(
            expiry_time=0.5,
            params=SVIParams(a=0.02, b=0.5, rho=-0.9, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=_forward(0.5, spot, r, q),
            n_quotes_total=5,
            n_quotes_used=5,
        )
        sl_high = FittedSlice(
            expiry_time=2.0,
            params=SVIParams(a=0.08, b=0.5, rho=-0.9, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=_forward(2.0, spot, r, q),
            n_quotes_total=5,
            n_quotes_used=5,
        )
        fs = FittedSurface(
            spot=spot,
            risk_free=r,
            div_yield=q,
            forward_curve=(
                (0.5, _forward(0.5, spot, r, q)),
                (2.0, _forward(2.0, spot, r, q)),
            ),
            fitted_slices=(sl_low, sl_high),
        )

        # K=50 on the steep smile drives the Dupire denominator NEGATIVE —
        # prove the mechanism directly, then confirm dupire_at maps it to nan.
        from arbfree_vol.pricing.local_vol import (
            _dw_dk, _d2w_dk2, _dupire_denominator,
        )
        from arbfree_vol.surface.interpolate import total_variance_at, _forward_at

        F_T = _forward_at(fs, 1.0)
        w = total_variance_at(fs, 50.0, 1.0)
        k = math.log(50.0 / F_T)
        dwdk = _dw_dk(fs, 50.0, 1.0, F_T)
        d2w = _d2w_dk2(fs, 50.0, 1.0, F_T)
        den = _dupire_denominator(w, k, dwdk, d2w)
        assert den < 0.0, (
            f"expected negative Dupire denominator at K=50 on the steep "
            f"smile, got {den}"
        )

        val = dupire_at(fs, K=50.0, T=1.0)
        assert math.isnan(val)

        # Interior, well-behaved moneyness still evaluates.
        ok = dupire_at(fs, K=100.0, T=1.0)
        assert not math.isnan(ok)
        assert ok > 0.0

    def test_dupire_at_nan_propagated_from_dw_dk(self, monkeypatch) -> None:
        """When the moneyness derivative is nan (degenerate step), dupire_at
        propagates nan — the caller sees undefined, not a crash."""
        from arbfree_vol.pricing import local_vol as lv_mod

        fs = _flat_fitted_surface(0.5, 2.0, 0.2)
        monkeypatch.setattr(
            lv_mod, "_dw_dk", lambda *a, **k: math.nan
        )

        val = dupire_at(fs, K=100.0, T=1.0)
        assert math.isnan(val)

    def test_dupire_at_nan_through_nan_denominator(self, monkeypatch) -> None:
        """A nan denominator propagates to nan (nan comparisons are False,
        so neither the denominator nor the sigma_loc_sq guard trips; the
        sqrt of nan is nan).  Pins that the guard rails do not misclassify
        an undefined cell as a valid one."""
        from arbfree_vol.pricing import local_vol as lv_mod

        fs = _flat_fitted_surface(0.5, 2.0, 0.2)
        monkeypatch.setattr(
            lv_mod, "_dupire_denominator", lambda *a, **k: math.nan
        )

        val = dupire_at(fs, K=100.0, T=1.0)
        assert math.isnan(val)


# ---------------------------------------------------------------------------
# Test: dupire grid validation
# ---------------------------------------------------------------------------
class TestDupireGridValidation:
    """dupire validates the grid dimensions up front."""

    def test_dupire_requires_three_strikes(self) -> None:
        fs = _flat_fitted_surface(0.5, 2.0, 0.2)
        with pytest.raises(ValueError, match="at least 3 strikes"):
            dupire(fs, strikes=[90.0, 100.0], maturities=[0.5, 1.0, 1.5])

    def test_dupire_requires_three_maturities(self) -> None:
        fs = _flat_fitted_surface(0.5, 2.0, 0.2)
        with pytest.raises(ValueError, match="at least 3 maturities"):
            dupire(fs, strikes=[90.0, 100.0, 110.0], maturities=[0.5, 1.0])


# ---------------------------------------------------------------------------
# Test: sub-2-slice surface + fallback masking
# ---------------------------------------------------------------------------
class TestDupireSubTwoSliceFallbackMask:
    """A sub-2-slice surface cannot produce a Dupire time derivative.  The
    only accepted grid is one whose EVERY row is masked as a fallback
    maturity (all-nan, no evaluation); any other row must fail clearly."""

    def _single_slice_surface(self) -> FittedSurface:
        from arbfree_vol.svi.model import SVIParams
        from arbfree_vol.models.fitted import FittedSlice, FittedSurface

        spot = 100.0
        r = 0.05
        q = 0.0
        sl = FittedSlice(
            expiry_time=1.0,
            params=SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=_forward(1.0, spot, r, q),
            n_quotes_total=5,
            n_quotes_used=5,
        )
        return FittedSurface(
            spot=spot,
            risk_free=r,
            div_yield=q,
            forward_curve=((1.0, _forward(1.0, spot, r, q)),),
            fitted_slices=(sl,),
        )

    def test_single_slice_unmasked_raises(self) -> None:
        """A single-slice surface with an unmasked grid row raises the
        clear sub-2-slice error instead of leaking an obscure
        out-of-range ValueError."""
        fs = self._single_slice_surface()
        with pytest.raises(ValueError, match="at least 2 fitted slices"):
            dupire(fs, strikes=[90.0, 100.0, 110.0], maturities=[0.5, 1.0, 1.5])

    def test_single_slice_all_rows_masked_ok(self) -> None:
        """When EVERY grid row is masked as a fallback maturity the grid is
        all-nan and dupire returns without evaluating any cell."""
        fs = self._single_slice_surface()
        lv = dupire(
            fs,
            strikes=[90.0, 100.0, 110.0],
            maturities=[1.0, 1.0 + 1e-4, 1.0 + 2e-4],
            fallback_slices=[1.0],
        )
        assert len(lv.grid) == 3
        for row in lv.grid:
            assert all(math.isnan(v) for v in row)

    def test_single_slice_partial_mask_raises(self) -> None:
        """Only SOME rows masked on a single-slice surface still raises:
        the unmasked row would need evaluation."""
        fs = self._single_slice_surface()
        with pytest.raises(ValueError, match="at least 2 fitted slices"):
            dupire(
                fs,
                strikes=[90.0, 100.0, 110.0],
                maturities=[0.5, 1.0, 1.5],
                fallback_slices=[1.0],
            )


# ---------------------------------------------------------------------------
# Test: _eval_cell calendar-arb vs genuine out-of-range
# ---------------------------------------------------------------------------
class TestEvalCell:
    """_eval_cell maps a calendar-arbitrage ValueError to nan but re-raises
    genuine out-of-surface errors."""

    def test_eval_cell_calendar_arb_maps_to_nan(self) -> None:
        """A cell whose dw/dT <= 0 (calendar arbitrage) is marked undefined
        (nan), not fatal."""
        spot = 100.0
        r = 0.05
        q = 0.0
        # Later slice has LOWER total variance → w decreasing with T
        sl_low = FittedSlice(
            expiry_time=0.5,
            params=SVIParams(a=0.3 ** 2 * 0.5, b=0.0, rho=0.0, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=_forward(0.5, spot, r, q),
            n_quotes_total=5,
            n_quotes_used=5,
        )
        sl_high = FittedSlice(
            expiry_time=2.0,
            params=SVIParams(a=0.1 ** 2 * 2.0, b=0.0, rho=0.0, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=_forward(2.0, spot, r, q),
            n_quotes_total=5,
            n_quotes_used=5,
        )
        fs = FittedSurface(
            spot=spot,
            risk_free=r,
            div_yield=q,
            forward_curve=(
                (0.5, _forward(0.5, spot, r, q)),
                (2.0, _forward(2.0, spot, r, q)),
            ),
            fitted_slices=(sl_low, sl_high),
        )

        from arbfree_vol.pricing.local_vol import _eval_cell
        val = _eval_cell(fs, K=100.0, T=1.5, dT=1e-3)
        assert math.isnan(val)

    def test_eval_cell_out_of_range_reraised(self) -> None:
        """A genuinely out-of-range query (T below the surface) is
        re-raised, not silently nan."""
        from arbfree_vol.pricing.local_vol import _eval_cell

        fs = _flat_fitted_surface(0.5, 2.0, 0.2)
        with pytest.raises(ValueError, match="below"):
            _eval_cell(fs, K=100.0, T=0.001, dT=1e-3)


# ---------------------------------------------------------------------------
# Test: _fallback_precompute empty branch
# ---------------------------------------------------------------------------
class TestFallbackPrecompute:
    """_fallback_precompute short-circuits when no fallback slices are
    supplied."""

    def test_no_fallback_slices_returns_empty(self) -> None:
        from arbfree_vol.pricing.local_vol import _fallback_precompute
        fs = _flat_fitted_surface(0.5, 2.0, 0.2)
        fallback_set, fitted_times = _fallback_precompute(fs, None)
        assert fallback_set == set()
        assert fitted_times == ()
