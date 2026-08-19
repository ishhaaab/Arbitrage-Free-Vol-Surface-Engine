"""Tests for surface interpolation (FittedSurface, total_variance_at, iv_at)."""

import math

import pytest
from pytest import approx

from arbfree_vol.svi.model import SVIParams, svi_total_variance
from arbfree_vol.models.fitted import FittedSlice, FittedSurface
from arbfree_vol.surface.interpolate import (
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
        from arbfree_vol.repair.report import RepairReport, RepairMetrics

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


# ── Edge behaviour tests ─────────────────────────────────────────────────────


class TestInterpolationEdges:
    """Edge behaviour of ``total_variance_at`` / ``iv_at``: 2-D interior
    analytic values, strike extrapolation, single-expiry surfaces,
    duplicate maturities, and invalid inputs — all per the documented
    contract in ``total_variance_at``."""

    def _smile_surface(self) -> FittedSurface:
        """Two-slice surface with a non-flat smile (b != 0) so the
        log-moneyness matters per slice."""
        spot = 100.0
        r = 0.05
        q = 0.0
        T_low, T_high = 0.5, 2.0
        fwd_low = _forward(T_low, spot, r, q)
        fwd_high = _forward(T_high, spot, r, q)
        sl_low = FittedSlice(
            expiry_time=T_low,
            params=SVIParams(a=0.02, b=0.3, rho=-0.3, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=fwd_low,
            n_quotes_total=5,
            n_quotes_used=5,
        )
        sl_high = FittedSlice(
            expiry_time=T_high,
            params=SVIParams(a=0.08, b=0.3, rho=-0.3, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=fwd_high,
            n_quotes_total=5,
            n_quotes_used=5,
        )
        return FittedSurface(
            spot=spot,
            risk_free=r,
            div_yield=q,
            forward_curve=((T_low, fwd_low), (T_high, fwd_high)),
            fitted_slices=(sl_low, sl_high),
        )

    def test_2d_interior_matches_analytic_linear_interpolation(self) -> None:
        """An interior (K, T) point on a smiling surface: total variance
        is the documented theta-weighted average of the bracketing
        slices' own-forward SVI values (recomputed independently here),
        and iv is sqrt(w/T)."""
        fs = self._smile_surface()
        K, T = 105.0, 1.25
        T_low, T_high = 0.5, 2.0
        sl_low, sl_high = fs.fitted_slices[0], fs.fitted_slices[1]

        w_low = svi_total_variance(
            math.log(K / sl_low.forward_price),
            sl_low.params.a, sl_low.params.b, sl_low.params.rho,
            sl_low.params.m, sl_low.params.sigma,
        )
        w_high = svi_total_variance(
            math.log(K / sl_high.forward_price),
            sl_high.params.a, sl_high.params.b, sl_high.params.rho,
            sl_high.params.m, sl_high.params.sigma,
        )
        theta = (T - T_low) / (T_high - T_low)
        w_expected = w_low + theta * (w_high - w_low)

        assert total_variance_at(fs, K=K, T=T) == approx(w_expected, abs=1e-12)
        assert iv_at(fs, K=K, T=T) == approx(math.sqrt(w_expected / T), abs=1e-12)

    def test_flat_surface_interior_known_value(self) -> None:
        """Interior point on a flat surface: w = sigma^2 * T exactly,
        independent of strike."""
        sigma = 0.2
        fs = _flat_fitted_surface(sigma=sigma)
        assert total_variance_at(fs, K=105.0, T=1.25) == approx(
            sigma ** 2 * 1.25, abs=1e-12
        )
        assert iv_at(fs, K=105.0, T=1.25) == approx(sigma, abs=1e-12)

    def test_strike_extrapolation_uses_svi_wings(self) -> None:
        """Strikes far outside the fitted moneyness range are NOT
        rejected: the SVI smile is evaluated at log(K/F), so the value
        equals the smile's wing extrapolation at the slice (documented
        in ``total_variance_at``).

        The surface is NON-FLAT (``b=0.3, rho=-0.3``) on purpose: with
        ``b=0`` the smile is constant in log-moneyness, so a mutation
        that broke the ``log(K/F)`` moneyness handling in the wing path
        (e.g. using the spot or a wrong forward) could never be caught.
        On this surface the expected wing value is recomputed
        independently from the SVI formula at the slice's OWN forward,
        so any moneyness-handling mutation in the wing path changes the
        extrapolated value and FAILS the test."""
        fs = self._smile_surface()
        sl = fs.fitted_slices[0]  # T=0.5 slice, b=0.3, rho=-0.3
        K_far = 300.0
        w_direct = svi_total_variance(
            math.log(K_far / sl.forward_price),
            sl.params.a, sl.params.b, sl.params.rho,
            sl.params.m, sl.params.sigma,
        )
        assert total_variance_at(fs, K=K_far, T=sl.expiry_time) == approx(
            w_direct, abs=1e-10
        )

    def test_single_expiry_surface(self) -> None:
        """A single-expiry surface evaluates exactly at its expiry; every
        other T raises ValueError (out of range)."""
        sigma = 0.2
        T = 0.5
        fwd = _forward(T)
        sl = FittedSlice(
            expiry_time=T,
            params=SVIParams(a=sigma ** 2 * T, b=0.0, rho=0.0, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=fwd,
            n_quotes_total=5,
            n_quotes_used=5,
        )
        fs = FittedSurface(
            spot=100.0,
            risk_free=0.05,
            div_yield=0.0,
            forward_curve=((T, fwd),),
            fitted_slices=(sl,),
        )

        assert iv_at(fs, K=100.0, T=T) == approx(sigma, rel=1e-3)
        assert total_variance_at(fs, K=100.0, T=T) == approx(
            sigma ** 2 * T, rel=1e-3
        )

        with pytest.raises(ValueError, match="below the surface range"):
            total_variance_at(fs, K=100.0, T=0.1)
        with pytest.raises(ValueError, match="above the surface range"):
            total_variance_at(fs, K=100.0, T=1.0)

    def test_expiry_tolerance_snaps_to_boundary_slice(self) -> None:
        """Queries within ``_EXACT_EXPIRY_TOL`` of a boundary expiry are
        SNAPPED to that boundary slice instead of interpolating or
        extrapolating in expiry.

        Regression: the exact-match loop is strict (``<``), so a query
        at exactly ``T_min + tol`` / ``T_max - tol`` used to fall
        through to the interior interpolation path — a ``T_max + tol``
        query extrapolated PAST the last slice (theta > 1) and returned
        a total variance different from the boundary slice's own value.
        With the snap, every query the tolerance admits evaluates the
        boundary slice exactly.
        """
        from arbfree_vol.surface.interpolate import _EXACT_EXPIRY_TOL

        fs = self._smile_surface()
        K = 105.0
        sl_low, sl_high = fs.fitted_slices[0], fs.fitted_slices[1]

        w_low_exact = svi_total_variance(
            math.log(K / sl_low.forward_price),
            sl_low.params.a, sl_low.params.b, sl_low.params.rho,
            sl_low.params.m, sl_low.params.sigma,
        )
        w_high_exact = svi_total_variance(
            math.log(K / sl_high.forward_price),
            sl_high.params.a, sl_high.params.b, sl_high.params.rho,
            sl_high.params.m, sl_high.params.sigma,
        )

        # T_min +/- tol snaps to the T_min slice; T_max +/- tol snaps to
        # the T_max slice (the pre-fix values differed at the ~1e-11
        # level because they went through the interpolation path, so an
        # abs=1e-15 tolerance discriminates the snap).
        for T in [sl_low.expiry_time - _EXACT_EXPIRY_TOL,
                  sl_low.expiry_time + _EXACT_EXPIRY_TOL]:
            assert total_variance_at(fs, K=K, T=T) == approx(w_low_exact, abs=1e-15), (
                f"T={T} not snapped to the T_min slice"
            )
        for T in [sl_high.expiry_time - _EXACT_EXPIRY_TOL,
                  sl_high.expiry_time + _EXACT_EXPIRY_TOL]:
            assert total_variance_at(fs, K=K, T=T) == approx(w_high_exact, abs=1e-15), (
                f"T={T} not snapped to the T_max slice"
            )

    def test_duplicate_maturities_resolve_to_first_slice(self) -> None:
        """Duplicate expiries in fitted_slices: the exact-match loop
        returns on the FIRST slice with the matching expiry, so the
        first slice's smile wins (documented first-slice precedence)."""
        T = 0.5
        fwd = _forward(T)
        sl1 = FittedSlice(
            expiry_time=T,
            params=SVIParams(a=0.02, b=0.0, rho=0.0, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=fwd,
            n_quotes_total=5,
            n_quotes_used=5,
        )
        sl2 = FittedSlice(
            expiry_time=T,
            params=SVIParams(a=0.10, b=0.0, rho=0.0, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=fwd,
            n_quotes_total=5,
            n_quotes_used=5,
        )
        fs = FittedSurface(
            spot=100.0,
            risk_free=0.05,
            div_yield=0.0,
            forward_curve=((T, fwd),),
            fitted_slices=(sl1, sl2),
        )

        w = total_variance_at(fs, K=100.0, T=T)
        assert w == approx(sl1.params.a, abs=1e-12)
        assert w != approx(sl2.params.a, abs=1e-12)

    def test_queries_outside_all_expiries_use_interior_fallback(self) -> None:
        """A query T strictly between two well-separated expiries (but not
        within tol of either) is resolved by the interior linear
        interpolation — the documented 2-D fallback when no exact slice
        matches."""
        T_low = 0.5
        T_high = 2.0
        fwd_low = _forward(T_low)
        fwd_high = _forward(T_high)

        sl_low = FittedSlice(
            expiry_time=T_low,
            params=SVIParams(a=0.02, b=0.0, rho=0.0, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=fwd_low,
            n_quotes_total=5,
            n_quotes_used=5,
        )
        sl_high = FittedSlice(
            expiry_time=T_high,
            params=SVIParams(a=0.08, b=0.0, rho=0.0, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=fwd_high,
            n_quotes_total=5,
            n_quotes_used=5,
        )
        fs = FittedSurface(
            spot=100.0,
            risk_free=0.05,
            div_yield=0.0,
            forward_curve=((T_low, fwd_low), (T_high, fwd_high)),
            fitted_slices=(sl_low, sl_high),
        )

        # T=1.0 is strictly between 0.5 and 2.0 and not within tol of either.
        w = total_variance_at(fs, K=100.0, T=1.0)
        # theta = (1.0 - 0.5) / (2.0 - 0.5) = 1/3
        w_expected = 0.02 + (1.0 / 3.0) * (0.08 - 0.02)
        assert w == approx(w_expected, abs=1e-12)

    def test_invalid_inputs(self) -> None:
        """T=0, a negative-variance fit, and a non-positive strike each
        produce the documented behaviour: ValueError."""
        fs = _flat_fitted_surface()

        # T=0 is below the surface range.
        with pytest.raises(ValueError, match="below the surface range"):
            total_variance_at(fs, K=100.0, T=0.0)

        # Negative total variance: returned as-is by total_variance_at,
        # then rejected by iv_at (math domain error from sqrt).
        T = 0.5
        fwd = _forward(T)
        sl = FittedSlice(
            expiry_time=T,
            params=SVIParams(a=-0.05, b=0.0, rho=0.0, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=fwd,
            n_quotes_total=5,
            n_quotes_used=5,
        )
        fs_neg = FittedSurface(
            spot=100.0,
            risk_free=0.05,
            div_yield=0.0,
            forward_curve=((T, fwd),),
            fitted_slices=(sl,),
        )
        assert total_variance_at(fs_neg, K=100.0, T=T) < 0.0
        with pytest.raises(ValueError):
            iv_at(fs_neg, K=100.0, T=T)

        # Non-positive strike: math domain error from log(K/F).
        with pytest.raises(ValueError):
            total_variance_at(fs, K=-10.0, T=0.5)
        with pytest.raises(ValueError):
            total_variance_at(fs, K=0.0, T=0.5)
