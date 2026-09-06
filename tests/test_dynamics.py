"""Tests for the surface dynamics module (PCA on a total-variance grid)."""

from __future__ import annotations

from datetime import date, timedelta
import math

import numpy as np
import pytest

from arbfree_vol.models.option import OptionType, OptionContract, BlackScholesInput
from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.fitted import FittedSlice
from arbfree_vol.svi.model import SVIParams, svi_total_variance
from arbfree_vol.dynamics import (
    fit_surface_series,
    total_variance_matrix,
    pca_deformations,
    PCAResult,
    SurfaceSeries,
    SurfaceSnapshot,
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


def _snapshot_with_params(
    params_by_expiry: dict[float, SVIParams], day: date
) -> SurfaceSnapshot:
    """Build a SurfaceSnapshot directly from hand-set SVI parameters.

    Bypasses the repair pipeline so matrix tests can pin exact parameter
    vectors (including degenerate ones like b=0) without depending on
    optimizer outcomes.
    """
    slices = tuple(
        FittedSlice(
            expiry_time=expiry,
            params=p,
            rmse=0.0,
            forward_price=SPOT * math.exp(R * expiry),
            n_quotes_total=9,
            n_quotes_used=9,
        )
        for expiry, p in params_by_expiry.items()
    )
    return SurfaceSnapshot(snapshot_date=day, fitted_slices=slices)


# ---------------------------------------------------------------------------
# Tests for the total-variance grid matrix
# ---------------------------------------------------------------------------


class TestTotalVarianceMatrix:
    """Shape, values, and identifiability of total_variance_matrix()."""

    def test_matrix_shape_buckets_times_knots(self) -> None:
        """Two buckets on the default 13-knot grid -> shape (2, 26)."""
        p = SVIParams(a=0.04, b=0.3, rho=-0.4, m=0.0, sigma=0.25)
        series = SurfaceSeries((
            _snapshot_with_params({0.5: p, 1.0: p}, date(2030, 1, 1)),
            _snapshot_with_params({0.5: p, 1.0: p}, date(2030, 1, 2)),
        ))

        matrix, buckets, knots, labels = total_variance_matrix(series)

        assert matrix.shape == (2, 26)
        assert buckets == (0.5, 1.0)
        assert len(knots) == 13
        assert len(labels) == 26

    def test_matrix_entries_are_total_variance_on_grid(self) -> None:
        """Each entry is the fitted slice's w(k) at its (bucket, knot)."""
        p = SVIParams(a=0.04, b=0.3, rho=-0.4, m=0.0, sigma=0.25)
        series = SurfaceSeries((
            _snapshot_with_params({1.0: p}, date(2030, 1, 1)),
        ))
        grid = (-0.2, 0.0, 0.1)

        matrix, _, knots, labels = total_variance_matrix(series, k_grid=grid)

        expected = [svi_total_variance(k, p.a, p.b, p.rho, p.m, p.sigma) for k in grid]
        assert np.allclose(matrix[0], expected, rtol=1e-12)
        assert np.allclose(knots, grid)
        assert labels == ("1.000_k-0.20", "1.000_k+0.00", "1.000_k+0.10")

    def test_flat_smile_rows_invariant_to_parameter_vector(self) -> None:
        """b=0 is an exact SVI degeneracy: w(k) = a for every k, whatever
        (rho, m, sigma) say.  Two snapshots with identical flat smiles but
        wildly different parameter vectors must produce identical matrix
        rows — the property the w(k) basis exists to guarantee.  (The old
        raw-parameter matrix failed this: rho/m/sigma columns differed.)"""
        p1 = SVIParams(a=0.04, b=0.0, rho=-0.7, m=0.10, sigma=0.50)
        p2 = SVIParams(a=0.04, b=0.0, rho=0.30, m=-0.20, sigma=0.90)
        series = SurfaceSeries((
            _snapshot_with_params({1.0: p1}, date(2030, 1, 1)),
            _snapshot_with_params({1.0: p2}, date(2030, 1, 2)),
        ))

        matrix, _, _, _ = total_variance_matrix(series)

        assert np.allclose(matrix[0], matrix[1])
        assert np.allclose(matrix[0], 0.04)

    def test_missing_slice_rows_are_nan(self) -> None:
        """A snapshot without a bucket gets NaN across that bucket's knots;
        present buckets are NaN-free."""
        p = SVIParams(a=0.04, b=0.3, rho=-0.4, m=0.0, sigma=0.25)
        series = SurfaceSeries((
            _snapshot_with_params({0.5: p, 1.0: p}, date(2030, 1, 1)),
            _snapshot_with_params({0.5: p}, date(2030, 1, 2)),
        ))

        matrix, _, knots, _ = total_variance_matrix(series)
        long_cols = slice(len(knots), 2 * len(knots))

        assert np.all(np.isnan(matrix[1, long_cols]))
        assert not np.any(np.isnan(matrix[0]))

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
        matrix, buckets, knots, labels = total_variance_matrix(series)

        # Long-dated columns (bucket ~1.0) should be NaN in row 1
        long_cols = slice(len(knots), 2 * len(knots))
        assert np.all(np.isnan(matrix[1, long_cols])), (
            "Row 1 (missing long slice) should be NaN for long-dated columns"
        )

        # PCA should not crash
        result = pca_deformations(matrix, n_components=2)
        assert isinstance(result, PCAResult)
        # n_features should be 26 (no column dropped since nan_frac = 0.5
        # is not > _NAN_DROP_THRESHOLD)
        assert result.n_features == 26

    def test_labels_encode_bucket_then_knot(self) -> None:
        """Labels follow '{bucket:.3f}_k{knot:+.2f}', knots ascending."""
        p = SVIParams(a=0.04, b=0.3, rho=-0.4, m=0.0, sigma=0.25)
        series = SurfaceSeries((
            _snapshot_with_params({0.5: p, 1.0: p}, date(2030, 1, 1)),
        ))
        _, _, knots, labels = total_variance_matrix(series)

        assert labels[0] == "0.500_k-0.30"
        assert labels[1] == "0.500_k-0.25"
        assert labels[12] == "0.500_k+0.30"
        assert labels[13] == "1.000_k-0.30"


class TestStandardization:
    """Behaviour of pca_deformations' standardize flag."""

    def test_standardized_pca_equals_manual_zscore_pca(self) -> None:
        """standardize=True on M is exactly PCA on the manually
        z-scored matrix with standardize=False (same SVD, same signs)."""
        rng = np.random.RandomState(5)
        scales = np.array([1.0, 10.0, 100.0, 1000.0])
        matrix = rng.normal(size=(12, 4)) * scales

        result = pca_deformations(matrix, n_components=3, standardize=True)
        z = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        manual = pca_deformations(z, n_components=3, standardize=False)

        assert np.allclose(np.asarray(result.scores), np.asarray(manual.scores))
        for c1, c2 in zip(result.components, manual.components):
            assert np.allclose(c1, c2)

    def test_standardization_stops_long_tenor_dominance(self) -> None:
        """Two buckets drifting by very different absolute amounts.

        Without scaling, PC1's loading mass sits almost entirely on the
        large-drift (long-dated) bucket — a pure scale artifact, not
        surface structure.  With standardize=True (default) both buckets
        are unit-variance, so a single common factor splits its loadings
        evenly across them.
        """
        snaps = []
        for i in range(12):
            a1 = 0.010 + 0.010 * i / 11   # short bucket drifts 0.010 total
            a2 = 0.050 + 0.100 * i / 11   # long bucket drifts 0.100 total
            p1 = SVIParams(a=a1, b=0.10, rho=-0.2, m=0.0, sigma=0.30)
            p2 = SVIParams(a=a2, b=0.10, rho=-0.2, m=0.0, sigma=0.30)
            day = date(2030, 1, 1) + timedelta(days=i)
            snaps.append(_snapshot_with_params({0.25: p1, 2.0: p2}, day))
        series = SurfaceSeries(tuple(snaps))

        matrix, _, knots, _ = total_variance_matrix(series)
        nk = len(knots)

        def short_bucket_share(comp: np.ndarray) -> float:
            comp = np.asarray(comp)
            return float(np.abs(comp[:nk]).sum() / np.abs(comp).sum())

        raw = pca_deformations(matrix, n_components=1, standardize=False)
        std = pca_deformations(matrix, n_components=1, standardize=True)

        assert short_bucket_share(raw.components[0]) < 0.10, (
            "unstandardized PC1 must be dominated by the long bucket"
        )
        assert 0.40 < short_bucket_share(std.components[0]) < 0.60, (
            "standardized PC1 must split loadings across both buckets"
        )


# ---------------------------------------------------------------------------
# Tests for PCA
# ---------------------------------------------------------------------------


class TestPCA:
    """Properties of the PCA decomposition."""

    def test_pca_cap_is_dimensional_not_rank(self) -> None:
        """The component cap is ``min(n_components, n_features,
        n_snapshots - 1)`` — a DIMENSIONAL cap, not a rank cap.

        The 10 x 4 input below has rank 2, so a rank-based cap would
        return at most 2 components.  The code instead caps by matrix
        dimensions (``min(request, 4, 9)``):
        - requesting more than the cap returns EXACTLY the cap (4);
        - requesting the cap returns exactly the cap (4) — two of the
          four components have zero variance because the rank is 2;
        - requesting below the cap returns exactly the request (2).
        """
        rng = np.random.RandomState(7)
        basis = rng.normal(size=(2, 4))   # two independent directions
        weights = rng.normal(size=(10, 2))
        matrix = weights @ basis          # shape (10, 4), rank 2
        assert np.linalg.matrix_rank(matrix) == 2

        cap = min(10 - 1, 4)              # n_snapshots - 1, n_features
        for requested in (7, 20, cap + 3):
            result = pca_deformations(matrix, n_components=requested)
            assert len(result.components) == cap, (
                f"requesting {requested} > cap must return exactly "
                f"min(requested, rows, cols) = {cap}, got "
                f"{len(result.components)}"
            )
            assert len(result.explained_variance_ratio) == cap
            assert all(len(s) == cap for s in result.scores)

        result_at_cap = pca_deformations(matrix, n_components=cap)
        assert len(result_at_cap.components) == cap
        # The extra components beyond the rank carry zero variance.
        assert result_at_cap.explained_variance_ratio[2] == pytest.approx(0.0, abs=1e-20)

        result_below = pca_deformations(matrix, n_components=2)
        assert len(result_below.components) == 2
        assert len(result_below.explained_variance_ratio) == 2
        assert all(len(s) == 2 for s in result_below.scores)

    def test_pca_known_rank_reconstruction(self) -> None:
        """A genuinely rank-2 matrix reconstructs exactly with k=2 via the
        API's own reconstruction path (``scores @ components``) and does
        NOT match with k=1 — proving two meaningful components.

        The matrix is built as the sum of two outer products, so its
        centred form lives in a 2-D subspace.  The module returns
        components (rows of ``Vt``) and scores (``U * S``); the intended
        reconstruction is ``scores @ components``, which equals the
        centred matrix when k >= rank.
        """
        rng = np.random.RandomState(11)
        v1 = rng.normal(size=4)
        v2 = rng.normal(size=4)
        u1 = rng.normal(size=10)
        u2 = rng.normal(size=10)
        matrix = np.outer(u1, v1) + np.outer(u2, v2)
        assert np.linalg.matrix_rank(matrix) == 2

        centred = matrix - matrix.mean(axis=0)

        result_k2 = pca_deformations(matrix, n_components=2, standardize=False)
        recon_k2 = np.asarray(result_k2.scores) @ np.vstack(result_k2.components)
        assert np.allclose(recon_k2, centred, atol=1e-10), (
            "k=2 reconstruction must reproduce the centred matrix"
        )

        result_k1 = pca_deformations(matrix, n_components=1, standardize=False)
        recon_k1 = np.asarray(result_k1.scores) @ np.vstack(result_k1.components)
        assert not np.allclose(recon_k1, centred, atol=1e-10), (
            "k=1 reconstruction must NOT match: the matrix has two "
            "meaningful components"
        )

    def test_pca_deterministic_output(self) -> None:
        """Two calls with identical input return identical arrays."""
        rng = np.random.RandomState(23)
        matrix = rng.normal(size=(8, 6))
        result_1 = pca_deformations(matrix, n_components=3)
        result_2 = pca_deformations(matrix, n_components=3)

        assert len(result_1.components) == len(result_2.components) == 3
        for c1, c2 in zip(result_1.components, result_2.components):
            assert np.array_equal(c1, c2)
        assert result_1.explained_variance_ratio == result_2.explained_variance_ratio
        assert result_1.scores == result_2.scores

    def test_pca_degenerate_inputs_return_empty_result(self) -> None:
        """Empty and all-NaN matrices follow the documented degenerate
        contract: an empty PCAResult with no components."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)

            # Empty (0 rows): zero snapshots, no columns survive the drop.
            empty = pca_deformations(np.empty((0, 4)), n_components=3)
            assert empty.n_snapshots == 0
            assert len(empty.components) == 0
            assert empty.explained_variance_ratio == ()
            assert empty.scores == ()

            # All-NaN (5 rows): every column is dropped by the 50 % NaN
            # rule -> zero retained features, one empty score per row.
            nan_matrix = pca_deformations(
                np.full((5, 4), np.nan), n_components=3
            )
            assert nan_matrix.n_snapshots == 5
            assert nan_matrix.n_features == 0
            assert len(nan_matrix.components) == 0
            assert nan_matrix.scores == ((), (), (), (), ())

    def test_single_parameter_drift_first_component_dominates(self) -> None:
        """A drift in rho across 20 snapshots yields a dominant first PC.

        Rho moves linearly from -0.5 to -0.1 while all other SVI params
        stay fixed.  The first component should capture >90 % of variance.

        (The threshold is 0.90, not 0.95: on the w(k) grid every knot
        carries the calibration noise of all five parameters, so PC2
        holds a few percent of systematic fit-tolerance residue that the
        old raw-parameter basis hid inside its near-constant columns.
        Dominance is the claim, not purity.)
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
        matrix, _, _, _ = total_variance_matrix(series)
        result = pca_deformations(matrix, n_components=3)

        assert result.explained_variance_ratio[0] > 0.90, (
            f"First component explains {result.explained_variance_ratio[0]:.4f}, "
            f"expected > 0.90"
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
        matrix, _, _, _ = total_variance_matrix(series)
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
