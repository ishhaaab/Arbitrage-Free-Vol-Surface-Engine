"""Tests for the surface dynamics module (PCA on SVI parameter time-series)."""

from __future__ import annotations

from datetime import date
import math

import numpy as np
import pytest

from arbfree_vol.models.option import OptionType, OptionContract, BlackScholesInput
from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.svi.model import SVIParams, svi_total_variance
from arbfree_vol.dynamics import (
    fit_surface_series,
    parameter_matrix,
    pca_deformations,
    PCAResult,
    _expiry_buckets,
    principal_mode_labels,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPOT = 100.0
R = 0.05
Q = 0.0
_DUMMY_DATE = date(2030, 1, 1)


# ---------------------------------------------------------------------------
# Helpers — build synthetic VolSurfaces from known SVI parameters
# ---------------------------------------------------------------------------


def _bs_price(otype: OptionType, strike: float, sigma: float, tt: float) -> float:
    """Black-Scholes price for a single option."""
    contract = OptionContract(
        symbol="X",
        option_type=otype,
        strike=strike,
        expiry_date=_DUMMY_DATE,
    )
    model = BlackScholesInput(
        contract=contract,
        spot=SPOT,
        expiry_time=tt,
        risk_free=R,
        div_yield=Q,
        volatility=sigma,
    )
    from arbfree_vol.pricing.black_scholes import price as bs_price

    return bs_price(model)


def _surface_from_svi_params(
    params: SVIParams, expiry: float = 1.0, n_strikes: int = 9
) -> VolSurface:
    """Build a VolSurface whose quotes are priced from the given SVI smile.

    For each strike, the SVI total variance is computed and converted to a
    Black-Scholes volatility (sigma = sqrt(w / T)), which is then used to
    price both a call and a put.
    """
    strikes = [
        SPOT * (1 + 0.1 * (i - n_strikes // 2)) for i in range(n_strikes)
    ]
    F = SPOT * math.exp((R - Q) * expiry)

    quotes: list[Quote] = []
    for K in strikes:
        k = math.log(K / F)
        w = svi_total_variance(k, params.a, params.b, params.rho, params.m, params.sigma)
        sigma_bs = math.sqrt(max(w / expiry, 1e-10))
        quotes.append(
            Quote(
                strike=K,
                option_type=OptionType.CALL,
                price=_bs_price(OptionType.CALL, K, sigma_bs, expiry),
            )
        )
        quotes.append(
            Quote(
                strike=K,
                option_type=OptionType.PUT,
                price=_bs_price(OptionType.PUT, K, sigma_bs, expiry),
            )
        )

    return VolSurface(
        spot=SPOT,
        risk_free=R,
        div_yield=Q,
        slices=[ExpirySlice(expiry_time=expiry, quotes=quotes)],
    )


def _surface_from_svi_params_multi(
    params_by_expiry: dict[float, SVIParams], n_strikes: int = 9
) -> VolSurface:
    """Build a VolSurface with multiple expiry slices, each from its own SVI."""
    slices: list[ExpirySlice] = []
    for expiry, params in params_by_expiry.items():
        strikes = [
            SPOT * (1 + 0.1 * (i - n_strikes // 2)) for i in range(n_strikes)
        ]
        F = SPOT * math.exp((R - Q) * expiry)
        quotes: list[Quote] = []
        for K in strikes:
            k = math.log(K / F)
            w = svi_total_variance(
                k, params.a, params.b, params.rho, params.m, params.sigma
            )
            sigma_bs = math.sqrt(max(w / expiry, 1e-10))
            quotes.append(
                Quote(
                    strike=K,
                    option_type=OptionType.CALL,
                    price=_bs_price(OptionType.CALL, K, sigma_bs, expiry),
                )
            )
            quotes.append(
                Quote(
                    strike=K,
                    option_type=OptionType.PUT,
                    price=_bs_price(OptionType.PUT, K, sigma_bs, expiry),
                )
            )
        slices.append(ExpirySlice(expiry_time=expiry, quotes=quotes))
    return VolSurface(spot=SPOT, risk_free=R, div_yield=Q, slices=slices)


# ---------------------------------------------------------------------------
# Tests for parameter matrix shape and NaN handling
# ---------------------------------------------------------------------------


class TestParameterMatrix:
    """Shape and missing-data behaviour of parameter_matrix()."""

    def test_parameter_matrix_consistent_shape(self) -> None:
        """Two snapshots each with two expiry buckets -> shape (2, 2*5)."""
        base_params = SVIParams(a=0.0, b=0.3, rho=-0.4, m=0.0, sigma=0.25)

        surf1 = _surface_from_svi_params_multi(
            {0.5: base_params, 1.0: base_params}
        )
        surf2 = _surface_from_svi_params_multi(
            {0.5: base_params, 1.0: base_params}
        )

        surfaces: list[tuple[date, VolSurface]] = [
            (date(2030, 1, 1), surf1),
            (date(2030, 1, 2), surf2),
        ]
        series = fit_surface_series(surfaces)
        matrix, buckets, labels = parameter_matrix(series)

        # Two buckets -> 10 features
        assert matrix.shape == (2, 10), f"Expected (2, 10), got {matrix.shape}"

    def test_expiry_buckets_union(self) -> None:
        """_expiry_buckets returns the union of all expiries across snapshots."""
        base_params = SVIParams(a=0.0, b=0.3, rho=-0.4, m=0.0, sigma=0.25)

        surf1 = _surface_from_svi_params_multi({0.5: base_params})
        surf2 = _surface_from_svi_params_multi({0.5: base_params, 1.0: base_params})
        surf3 = _surface_from_svi_params_multi({1.5: base_params})

        surfaces: list[tuple[date, VolSurface]] = [
            (date(2030, 1, 1), surf1),
            (date(2030, 1, 2), surf2),
            (date(2030, 1, 3), surf3),
        ]
        series = fit_surface_series(surfaces)
        buckets = _expiry_buckets(series.snapshots)

        assert 0.5 in buckets
        assert 1.0 in buckets
        assert 1.5 in buckets

    def test_missing_slice_nan_imputation_pca_does_not_crash(self) -> None:
        """Missing slices produce NaN in the matrix; PCA handles it gracefully.

        Snapshot 1 has both short and long slices, snapshot 2 has only the
        short slice.  The long-dated columns are 50 % NaN (1 of 2 rows).
        Since 0.5 is not *more* than _NAN_DROP_THRESHOLD, the column is
        kept and NaN is imputed with the column mean.
        """
        base_params = SVIParams(a=0.0, b=0.3, rho=-0.4, m=0.0, sigma=0.25)

        surf1 = _surface_from_svi_params_multi({0.5: base_params, 1.0: base_params})
        surf2 = _surface_from_svi_params_multi({0.5: base_params})

        surfaces: list[tuple[date, VolSurface]] = [
            (date(2030, 1, 1), surf1),
            (date(2030, 1, 2), surf2),
        ]
        series = fit_surface_series(surfaces)
        matrix, buckets, labels = parameter_matrix(series)

        # Long-dated columns (bucket ~1.0) should be NaN in row 1
        # Index: 0-4 are bucket 0.5, 5-9 are bucket 1.0
        long_cols = slice(5, 10)
        assert np.all(np.isnan(matrix[1, long_cols])), (
            "Row 1 (missing long slice) should be NaN for long-dated columns"
        )

        # PCA should not crash
        result = pca_deformations(matrix, n_components=2)
        assert isinstance(result, PCAResult)
        # n_features should be 10 (no column dropped since nan_frac = 0.5
        # is not > _NAN_DROP_THRESHOLD)
        assert result.n_features == 10

    def test_param_labels_format(self) -> None:
        """Labels follow the pattern '{bucket:.3f}_{param}'."""
        base_params = SVIParams(a=0.0, b=0.3, rho=-0.4, m=0.0, sigma=0.25)
        surf = _surface_from_svi_params_multi({0.5: base_params, 1.0: base_params})
        series = fit_surface_series([(date(2030, 1, 1), surf)])
        _, _, labels = parameter_matrix(series)

        assert labels[0] == "0.500_a"
        assert labels[1] == "0.500_b"
        assert labels[2] == "0.500_rho"
        assert labels[3] == "0.500_m"
        assert labels[4] == "0.500_sigma"
        assert labels[5] == "1.000_a"


# ---------------------------------------------------------------------------
# Tests for PCA
# ---------------------------------------------------------------------------


class TestPCA:
    """Properties of the PCA decomposition."""

    def test_pca_returns_requested_n_components(self) -> None:
        """PCA returns exactly the requested count, capped by the SVD
        dimensions ``min(n_components, n_features, n_snapshots - 1)``.

        The input is a deterministic known-rank matrix (6 snapshots x 10
        features, rank 2), so the returned counts are exact and cannot
        drift with calibration noise:
        - a request above the rank AND above the ``n_snapshots - 1`` cap
          returns exactly ``min(7, 10, 5) = 5`` components;
        - a request below the rank returns exactly that count.
        """
        rng = np.random.RandomState(7)
        basis = rng.normal(size=(2, 10))      # two independent directions
        weights = rng.normal(size=(6, 2))     # each row is a combination
        matrix = weights @ basis              # shape (6, 10), rank 2

        result_7 = pca_deformations(matrix, n_components=7)
        assert len(result_7.components) == 5, (
            f"expected exactly min(7, 10, 5) = 5 components, got "
            f"{len(result_7.components)}"
        )
        assert len(result_7.explained_variance_ratio) == 5
        assert len(result_7.scores[0]) == 5
        # Only the two nonzero singular values explain variance.
        assert result_7.explained_variance_ratio[2] == pytest.approx(0.0, abs=1e-20)
        assert result_7.explained_variance_ratio[3] == pytest.approx(0.0, abs=1e-20)

        result_1 = pca_deformations(matrix, n_components=1)
        assert len(result_1.components) == 1
        assert len(result_1.explained_variance_ratio) == 1
        assert len(result_1.scores[0]) == 1

    def test_single_parameter_drift_first_component_dominates(self) -> None:
        """A drift in rho across 20 snapshots yields a dominant first PC.

        Rho moves linearly from -0.5 to -0.1 while all other SVI params
        stay fixed.  The first component should capture >95 % of variance.
        """
        from datetime import timedelta

        base = dict(a=0.0, b=0.3, m=0.0, sigma=0.3)
        rhos = np.linspace(-0.5, -0.1, 20)
        base_dt = date(2030, 1, 1)

        surfaces: list[tuple[date, VolSurface]] = []
        for i, rho_val in enumerate(rhos):
            p = SVIParams(rho=float(rho_val), **base)
            surfaces.append(
                (base_dt + timedelta(days=i), _surface_from_svi_params(p))
            )

        series = fit_surface_series(surfaces)
        matrix, _, _ = parameter_matrix(series)
        result = pca_deformations(matrix, n_components=3)

        assert result.explained_variance_ratio[0] > 0.95, (
            f"First component explains {result.explained_variance_ratio[0]:.4f}, "
            f"expected > 0.95"
        )

    def test_two_parameter_rotation_two_components_dominate(self) -> None:
        """Oscillating rho and sigma produce two dominant components.

        Over 50 snapshots rho = 0.2 * cos(t) and sigma = 0.15 * sin(t) + 0.3
        sweep a full cycle.  The first two components should explain >80 %
        of cumulative variance.
        """
        t_vals = np.linspace(0, 2 * np.pi, 50)
        base = dict(a=0.0, b=0.3, m=0.0)
        fixed_sigma = 0.15

        from datetime import timedelta

        base_dt = date(2030, 1, 1)
        surfaces: list[tuple[date, VolSurface]] = []
        for i, t in enumerate(t_vals):
            rho_val = 0.2 * np.cos(t)
            sigma_val = fixed_sigma * np.sin(t) + 0.30
            p = SVIParams(rho=float(rho_val), sigma=float(sigma_val), **base)
            surfaces.append(
                (base_dt + timedelta(days=i), _surface_from_svi_params(p))
            )

        series = fit_surface_series(surfaces)
        matrix, _, _ = parameter_matrix(series)
        result = pca_deformations(matrix, n_components=5)

        cumul = sum(result.explained_variance_ratio[:2])
        assert cumul > 0.80, (
            f"First two components explain {cumul:.4f}, expected > 0.80"
        )

    def test_pca_single_snapshot_returns_empty_result(self) -> None:
        """A single observation yields no principal components."""
        matrix = np.array([[0.04, 0.4, -0.4, 0.05, 0.15]])
        result = pca_deformations(matrix, n_components=3)

        assert result.n_snapshots == 1
        assert result.n_features == 5
        assert len(result.components) == 0
        assert len(result.explained_variance_ratio) == 0
        assert result.scores == ((),)


# ---------------------------------------------------------------------------
# Tests for principal_mode_labels
# ---------------------------------------------------------------------------


class TestModeLabels:
    def test_labels_heuristic(self) -> None:
        assert principal_mode_labels(1) == ["Level"]
        assert principal_mode_labels(2) == ["Level", "Tilt"]
        assert principal_mode_labels(3) == ["Level", "Tilt", "Curvature"]

    def test_labels_beyond_three(self) -> None:
        labels = principal_mode_labels(5)
        assert len(labels) == 5
        assert labels[3] == "Mode 4"
        assert labels[4] == "Mode 5"


# ---------------------------------------------------------------------------
# Test for SurfaceSeries ordering
# ---------------------------------------------------------------------------


class TestSurfaceSeries:
    def test_snapshots_sorted_by_date(self) -> None:
        """SurfaceSeries must sort snapshots by date ascending."""
        base_params = SVIParams(a=0.0, b=0.3, rho=-0.4, m=0.0, sigma=0.25)

        surfaces: list[tuple[date, VolSurface]] = [
            (date(2030, 3, 1), _surface_from_svi_params(base_params)),
            (date(2030, 1, 1), _surface_from_svi_params(base_params)),
            (date(2030, 2, 1), _surface_from_svi_params(base_params)),
        ]
        series = fit_surface_series(surfaces)
        dates = [sn.snapshot_date for sn in series.snapshots]
        assert dates == sorted(dates)
