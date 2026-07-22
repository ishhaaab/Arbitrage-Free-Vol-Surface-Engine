"""Tests for Dupire local volatility (pricing/local_vol.py).

All tests are synthetic — no network, no yfinance.
"""

import math

import pytest
from pytest import approx

from arbfree_vol.svi.model import SVIParams, svi_total_variance
from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.surface.interpolate import (
    FittedSurface,
    total_variance_at,
    iv_at,
)
from arbfree_vol.pricing.local_vol import (
    LocalVolSurface,
    dupire_at,
    dupire,
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
