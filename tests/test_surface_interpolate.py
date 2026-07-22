"""Tests for surface interpolation (FittedSurface, total_variance_at, iv_at)."""

import math

import pytest
from pytest import approx

from arbfree_vol.svi.model import SVIParams, svi_total_variance
from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.surface.interpolate import (
    FittedSurface,
    build_fitted_surface,
    total_variance_at,
    iv_at,
)


def _forward(T: float, spot: float = 100.0, r: float = 0.05, q: float = 0.0) -> float:
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
        params=SVIParams(a=sigma ** 2 * T_low, b=0.0, rho=0.0, m=0.0, sigma=0.2),
        rmse=0.0,
        forward_price=_forward(T_low, spot, r, q),
        n_quotes_total=5,
        n_quotes_used=5,
    )
    sl_high = FittedSlice(
        expiry_time=T_high,
        params=SVIParams(a=sigma ** 2 * T_high, b=0.0, rho=0.0, m=0.0, sigma=0.2),
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


# ── iv_at tests ─────────────────────────────────────────────────────────────


class TestIvAt:
    """Tests for ``iv_at`` on a flat fitted surface."""

    def test_iv_at_flat_surface(self) -> None:
        """iv_at returns the flat sigma for any in-range T."""
        sigma = 0.2
        spot = 100.0
        fs = _flat_fitted_surface(sigma=sigma, spot=spot)

        for T in [0.5, 1.0, 1.5, 2.0]:
            result = iv_at(fs, K=spot, T=T)
            assert result == approx(sigma, rel=1e-3), f"Failed at T={T}"

    def test_iv_at_interior_interpolation(self) -> None:
        """Linear interpolation in w-space preserves flat sigma."""
        sigma = 0.2
        spot = 100.0
        fs = _flat_fitted_surface(sigma=sigma, spot=spot)

        T_interp = 1.25
        w = total_variance_at(fs, K=spot, T=T_interp)
        # For flat smile at sigma=0.2, w = sigma² × T = 0.04 × 1.25 = 0.05
        assert w == approx(sigma ** 2 * T_interp, rel=1e-3)
        # iv = sqrt(w / T) = sigma
        assert iv_at(fs, K=spot, T=T_interp) == approx(sigma, rel=1e-3)

    def test_iv_at_raises_out_of_surface(self) -> None:
        """T below the earliest slice expiry raises ValueError."""
        fs = _flat_fitted_surface(T_low=0.5, T_high=2.0)
        with pytest.raises(ValueError, match="below the surface range"):
            iv_at(fs, K=100.0, T=0.1)

    def test_iv_at_raises_above_surface(self) -> None:
        """T above the latest slice expiry raises ValueError."""
        fs = _flat_fitted_surface(T_low=0.5, T_high=2.0)
        with pytest.raises(ValueError, match="above the surface range"):
            iv_at(fs, K=100.0, T=3.0)


# ── total_variance_at tests ──────────────────────────────────────────────────


class TestTotalVarianceAt:
    """Tests for ``total_variance_at``."""

    def test_total_variance_at_matches_slice_at_exact_expiry(self) -> None:
        """At exact slice expiry, total_variance_at must equal the direct
        SVI evaluation."""
        sigma = 0.2
        spot = 100.0
        fs = _flat_fitted_surface(sigma=sigma, spot=spot)

        T = 0.5
        K = 110.0

        w_direct = total_variance_at(fs, K=K, T=T)

        # Direct SVI evaluation for the T=0.5 slice
        sl = fs.fitted_slices[0]  # T=0.5
        F = sl.forward_price
        k = math.log(K / F)
        w_expected = svi_total_variance(
            k, sl.params.a, sl.params.b, sl.params.rho,
            sl.params.m, sl.params.sigma,
        )

        assert w_direct == approx(w_expected, abs=1e-10)

    def test_total_variance_at_uses_own_forward_per_slice(self) -> None:
        """Each slice uses its own forward price (term structure test)."""
        # Two slices with slightly different SVI parameters such that the
        # forward difference matters.  We use non-zero b so the
        # log-moneyness matters.
        spot = 100.0
        r = 0.05
        q = 0.0
        T_low = 0.5
        T_high = 2.0

        forward_low = _forward(T_low, spot, r, q)
        forward_high = _forward(T_high, spot, r, q)

        sl_low = FittedSlice(
            expiry_time=T_low,
            params=SVIParams(a=0.02, b=0.3, rho=-0.3, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=forward_low,
            n_quotes_total=5,
            n_quotes_used=5,
        )
        sl_high = FittedSlice(
            expiry_time=T_high,
            params=SVIParams(a=0.08, b=0.3, rho=-0.3, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=forward_high,
            n_quotes_total=5,
            n_quotes_used=5,
        )

        fs = FittedSurface(
            spot=spot,
            risk_free=r,
            div_yield=q,
            forward_curve=((T_low, forward_low), (T_high, forward_high)),
            fitted_slices=(sl_low, sl_high),
        )

        # At exact slice expiry it should match direct evaluation.
        K = 105.0
        w = total_variance_at(fs, K=K, T=T_low)
        expected = svi_total_variance(
            math.log(K / forward_low),
            sl_low.params.a, sl_low.params.b, sl_low.params.rho,
            sl_low.params.m, sl_low.params.sigma,
        )
        assert w == approx(expected, abs=1e-10)


# ── build_fitted_surface tests ───────────────────────────────────────────────


class TestBuildFittedSurface:
    """Tests for ``build_fitted_surface`` from a RepairReport."""

    def test_build_fitted_surface_raises_on_no_cleaned_surface(self) -> None:
        """A RepairReport with cleaned_surface=None raises ValueError."""
        from arbfree_vol.arbitrage.report import ArbitrageReport
        from arbfree_vol.repair.report import RepairReport, RepairMetrics, RejectedQuote

        report = RepairReport(
            rejected=(),
            fitted_slices=(),
            remaining_violations=ArbitrageReport(violations=[]),
            metrics=RepairMetrics(
                n_rejected=0, n_total_quotes=0,
                n_slices_input=0, n_slices_fitted=0,
                n_violations_before=0, n_violations_after=0,
            ),
            cleaned_surface=None,
        )
        with pytest.raises(ValueError, match="no cleaned_surface"):
            build_fitted_surface(report)
