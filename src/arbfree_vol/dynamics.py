"""PCA-based surface dynamics analysis on time-series of fitted vol surfaces.

Provides tools to collect fitted surfaces across multiple snapshot dates,
evaluate each fitted smile on a fixed log-moneyness grid of total
variances, and perform PCA via SVD to identify the dominant modes of
deformation (Level, Tilt, Curvature).

The feature basis is total variance w(k) on a grid, NOT the raw SVI
parameters: SVI fits are non-unique (numerically distinct parameter
vectors can describe the same smile — the flat-smile case b=0 is an
exact example, and the round-trip calibration tests document the
near-degeneracy), so PCA over parameter coordinates mixes parametrization
noise into the modes.  w(k) on a fixed grid is the observable the smile
actually quotes: identical smiles produce identical rows no matter which
(a, b, rho, m, sigma) vector the optimizer happened to return, and all
three model families feed the grid through their common raw-SVI mapping.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from arbfree_vol.models.surface import VolSurface
from arbfree_vol.repair.engine import repair
from arbfree_vol.models.fitted import FittedSlice
from arbfree_vol.svi.model import svi_total_variance

_BUCKET_TOL = 1e-3
_NAN_DROP_THRESHOLD = 0.5


@dataclass(frozen=True, slots=True)
class SurfaceSnapshot:
    """One fitted vol surface at a point in time."""

    snapshot_date: date
    fitted_slices: tuple[FittedSlice, ...]


@dataclass(frozen=True, slots=True)
class SurfaceSeries:
    """Ordered collection of surface snapshots, sorted by date ascending."""

    snapshots: tuple[SurfaceSnapshot, ...]


@dataclass(frozen=True, slots=True)
class PCAResult:
    """Result of PCA decomposition on the feature matrix.

    Components are the principal directions in feature space (each is a
    length-n_features array).  Scores give the projection of each snapshot
    onto the component axes.
    """

    components: tuple[np.ndarray, ...]
    explained_variance_ratio: tuple[float, ...]
    scores: tuple[tuple[float, ...], ...]
    n_features: int
    n_snapshots: int


# ---------------------------------------------------------------------------
# Surface series construction
# ---------------------------------------------------------------------------


def fit_surface_series(surfaces: list[tuple[date, VolSurface]]) -> SurfaceSeries:
    """Fit each surface via the repair pipeline and build a sorted series.

    For each ``(snapshot_date, surface)`` pair, calls ``repair(surface)``
    and extracts ``report.fitted_slices``.

    Parameters
    ----------
    surfaces : list of (date, VolSurface)
        List of (observation date, market surface) pairs in any order.

    Returns
    -------
    SurfaceSeries
        Snapshots sorted by date ascending.
    """
    snapshots: list[SurfaceSnapshot] = []
    for snapshot_date, surface in surfaces:
        report = repair(surface)
        sn = SurfaceSnapshot(
            snapshot_date=snapshot_date,
            fitted_slices=tuple(report.fitted_slices),
        )
        snapshots.append(sn)
    snapshots.sort(key=lambda s: s.snapshot_date)
    return SurfaceSeries(tuple(snapshots))


# ---------------------------------------------------------------------------
# Expiry bucketing
# ---------------------------------------------------------------------------


def _expiry_buckets(snapshots: tuple[SurfaceSnapshot, ...]) -> tuple[float, ...]:
    """Union of all slice expiries across snapshots, rounded and sorted.

    Expiries are rounded to 3 decimal places and deduplicated within
    ``_BUCKET_TOL`` (1e-3).
    """
    buckets: set[float] = set()
    for sn in snapshots:
        for fs in sn.fitted_slices:
            buckets.add(round(fs.expiry_time, 3))
    return tuple(sorted(buckets))


# ---------------------------------------------------------------------------
# Total-variance grid matrix construction
# ---------------------------------------------------------------------------

# Default log-moneyness knots: 13 points spanning [-0.30, 0.30].  This
# brackets the liquid range of a typical US equity chain (a +/-20% strike
# span is k ~ +/-0.20); callers can pass any grid.
_DEFAULT_K_GRID: tuple[float, ...] = tuple(
    np.linspace(-0.30, 0.30, 13).tolist()
)


def total_variance_matrix(
    series: SurfaceSeries,
    k_grid: Sequence[float] = _DEFAULT_K_GRID,
) -> tuple[np.ndarray, tuple[float, ...], np.ndarray, tuple[str, ...]]:
    """Build the n_snapshots x n_features matrix of total variances w(k).

    Each feature is a fitted slice's total variance evaluated at one
    (expiry bucket, log-moneyness knot) pair.  Columns are ordered by
    bucket then knot ascending:

        bucket0_k-0.30, bucket0_k-0.25, ..., bucket1_k-0.30, ...

    Missing slices (present in some snapshots but not others) produce
    ``np.nan`` entries.

    Why this basis instead of raw SVI parameters: the SVI parametrization
    is non-identifiable (distinct parameter vectors can produce the same
    smile), so a parameter matrix is not a well-posed PCA object.  The
    w(k) grid is the observable, and identical smiles produce identical
    rows regardless of which parameter vector the optimizer returned.

    Parameters
    ----------
    series : SurfaceSeries
        The fitted surface series (from :func:`fit_surface_series`).
    k_grid : sequence of float
        Log-moneyness knots at which to evaluate each fitted smile.
        Defaults to 13 knots spanning [-0.30, 0.30].

    Returns
    -------
    matrix : np.ndarray, shape (n_snapshots, n_buckets * n_knots)
    expiry_buckets : tuple of float
        The unique expiry buckets (sorted).
    knots : np.ndarray
        The knots actually used, as a float array.
    labels : tuple of str
        Human-readable label for each column, e.g. ``"1.000_k-0.30"``.
    """
    knots = np.asarray(list(k_grid), dtype=float)
    buckets = _expiry_buckets(series.snapshots)

    labels: list[str] = []
    for b in buckets:
        for k in knots:
            labels.append(f"{b:.3f}_k{k:+.2f}")

    matrix = np.full((len(series.snapshots), len(labels)), np.nan)

    for i, sn in enumerate(series.snapshots):
        for j, bucket in enumerate(buckets):
            matched: FittedSlice | None = None
            for fs in sn.fitted_slices:
                if abs(fs.expiry_time - bucket) <= _BUCKET_TOL:
                    matched = fs
                    break
            if matched is None:
                continue
            p = matched.params
            base = j * len(knots)
            for kk, k in enumerate(knots):
                matrix[i, base + kk] = svi_total_variance(
                    float(k), p.a, p.b, p.rho, p.m, p.sigma
                )

    return matrix, buckets, knots, tuple(labels)


# ---------------------------------------------------------------------------
# PCA via SVD
# ---------------------------------------------------------------------------


def pca_deformations(
    matrix: np.ndarray, n_components: int = 3, *, standardize: bool = True
) -> PCAResult:
    """Perform PCA via SVD on the (n_snapshots × n_features) feature matrix.

    Handling of missing data
    ------------------------
    1. Columns whose NaN fraction exceeds ``_NAN_DROP_THRESHOLD`` (0.5, i.e.
       50 %) are dropped entirely.
    2. Remaining NaN values are imputed with the column mean (``np.nanmean``).
    3. Columns are centred by subtracting the (imputed) column mean.
    4. When ``standardize`` is True (default), columns are then scaled to
       unit variance (correlation PCA).  The w(k) grid mixes expiry buckets
       whose total-variance levels differ by an order of magnitude, so
       without scaling the long-dated buckets would dominate PC1 for scale
       reasons alone.  Zero-variance columns are left unscaled — they
       carry no information after centring.  With ``standardize=False``
       the PCA runs on the plain covariance matrix.

    PCA is then performed via ``numpy.linalg.svd`` on the centred (and
    optionally scaled) matrix.
    The number of components returned is capped at
    ``min(n_components, n_features_retained, n_snapshots - 1)``.

    The cap is purely dimensional — it depends only on the requested count
    and the matrix shape (columns that survive the NaN drop, rows minus
    one), NOT on the numerical rank of the input.  Requesting more
    components than the rank therefore returns the full dimensional cap,
    with trailing components carrying (approximately) zero explained
    variance.

    Degenerate inputs yield an empty :class:`PCAResult` (no components,
    empty ``explained_variance_ratio``, one empty score tuple per row):
    single-snapshot input (``n_snapshots < 2``) since no variance can be
    estimated from one observation, an all-NaN matrix (every column is
    dropped by the 50 % NaN rule), and an empty ``(0, n_features)`` matrix
    (zero rows; the mean-of-empty-slice column drop retains no columns).

    Parameters
    ----------
    matrix : np.ndarray, shape (n_snapshots, n_features)
    n_components : int
        Number of principal components to return (default 3).

    Returns
    -------
    PCAResult
    """
    # ---- Step 1: drop sparse columns ----
    nan_frac = np.isnan(matrix).mean(axis=0)
    keep = nan_frac <= _NAN_DROP_THRESHOLD
    X = matrix[:, keep].copy()

    # ---- Step 2: impute remaining NaN with column mean ----
    col_mean = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if len(inds[0]) > 0:
        X[inds] = col_mean[inds[1]]

    # ---- Step 3: centre, then optionally scale to unit variance ----
    X_centered = X - X.mean(axis=0)
    if standardize:
        col_std = X_centered.std(axis=0)
        col_std = np.where(col_std > 0.0, col_std, 1.0)
        X_centered = X_centered / col_std

    # ---- Step 4: SVD ----
    n_snapshots, n_features_retained = X_centered.shape
    n_comp = min(n_components, n_features_retained, n_snapshots - 1)

    if n_comp == 0:
        # Degenerate case: return empty result
        return PCAResult(
            components=(),
            explained_variance_ratio=(),
            scores=tuple(tuple() for _ in range(n_snapshots)),
            n_features=n_features_retained,
            n_snapshots=n_snapshots,
        )

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # explained variance ratio
    total_var = np.sum(S ** 2)
    if total_var <= 0.0:
        var_ratio = tuple(0.0 for _ in range(n_comp))
    else:
        var_ratio = tuple(float(S[i] ** 2 / total_var) for i in range(n_comp))

    # components = rows of Vt
    components = tuple(Vt[i, :].copy() for i in range(n_comp))

    # scores = U[:, :n_comp] * S[:n_comp]
    scores_matrix = U[:, :n_comp] * S[:n_comp]
    scores = tuple(
        tuple(float(scores_matrix[i, j]) for j in range(n_comp))
        for i in range(n_snapshots)
    )

    return PCAResult(
        components=components,
        explained_variance_ratio=var_ratio,
        scores=scores,
        n_features=n_features_retained,
        n_snapshots=n_snapshots,
    )


# ---------------------------------------------------------------------------
# Heuristic mode labels
# ---------------------------------------------------------------------------


def principal_mode_labels(n: int) -> list[str]:
    """Return heuristic labels for the first *n* principal components.

    The first three modes are conventionally named Level, Tilt, and
    Curvature — a heuristic borrowed from the interest-rate PCA
    literature (e.g. Cont-da-Fonseca-Durrleman).  Higher modes receive
    generic numeric labels.

    Parameters
    ----------
    n : int
        Number of labels requested.

    Returns
    -------
    list of str
    """
    base = ["Level", "Tilt", "Curvature"]
    if n <= 3:
        return base[:n]
    return base + [f"Mode {i}" for i in range(4, n + 1)]
