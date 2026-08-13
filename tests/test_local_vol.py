"""Tests for Dupire local volatility (pricing/local_vol.py).

All tests are synthetic — no network, no yfinance.
"""

import math

import pytest
from pytest import approx

from arbfree_vol.svi.model import SVIParams, svi_total_variance
from arbfree_vol.models.fitted import FittedSlice, FittedSurface
from arbfree_vol.surface.interpolate import (
    total_variance_at,
    iv_at,
)
from arbfree_vol.pricing.local_vol import (
    LocalVolSurface,
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
