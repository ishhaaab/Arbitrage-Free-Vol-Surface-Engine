"""PCA-based surface dynamics analysis on time-series of fitted SVI surfaces.

Provides tools to collect fitted SVI parameter surfaces across multiple
snapshot dates, build a parameter matrix, and perform PCA via SVD to
identify the dominant modes of deformation (Level, Tilt, Curvature).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from arbfree_vol.models.surface import VolSurface
from arbfree_vol.repair.engine import repair
from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.svi.model import SVIParams

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
    """Result of PCA decomposition on the parameter matrix.

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
# Parameter matrix construction
# ---------------------------------------------------------------------------


_PARAM_NAMES = ["a", "b", "rho", "m", "sigma"]


def parameter_matrix(
    series: SurfaceSeries,
) -> tuple[np.ndarray, tuple[float, ...], tuple[str, ...]]:
    """Build the n_snapshots × n_features parameter matrix.

    Each column corresponds to one parameter of one expiry bucket.  Columns
    are ordered by bucket then parameter name:

        bucket0_a, bucket0_b, bucket0_rho, bucket0_m, bucket0_sigma,
        bucket1_a, ...

    Missing slices (present in some snapshots but not others) produce
    ``np.nan`` entries.

    Parameters
    ----------
    series : SurfaceSeries
        The fitted surface series.

    Returns
    -------
    matrix : np.ndarray, shape (n_snapshots, n_features)
    expiry_buckets : tuple of float
        The unique expiry buckets (sorted).
    param_labels : tuple of str
        Human-readable label for each column, e.g. ``"1.000_a"``.
    """
    buckets = _expiry_buckets(series.snapshots)
    n_buckets = len(buckets)
    n_features = n_buckets * 5
    n_snapshots = len(series.snapshots)

    matrix = np.full((n_snapshots, n_features), np.nan)

    param_labels: list[str] = []
    for b in buckets:
        for pname in _PARAM_NAMES:
            param_labels.append(f"{b:.3f}_{pname}")

    for i, sn in enumerate(series.snapshots):
        for j, bucket in enumerate(buckets):
            matched: FittedSlice | None = None
            for fs in sn.fitted_slices:
                if abs(fs.expiry_time - bucket) <= _BUCKET_TOL:
                    matched = fs
                    break
            if matched is not None:
                base = j * 5
                matrix[i, base] = matched.params.a
                matrix[i, base + 1] = matched.params.b
                matrix[i, base + 2] = matched.params.rho
                matrix[i, base + 3] = matched.params.m
                matrix[i, base + 4] = matched.params.sigma

    return matrix, buckets, tuple(param_labels)


# ---------------------------------------------------------------------------
# PCA via SVD
# ---------------------------------------------------------------------------


def pca_deformations(
    matrix: np.ndarray, n_components: int = 3
) -> PCAResult:
    """Perform PCA via SVD on the (n_snapshots × n_features) parameter matrix.

    Handling of missing data
    ------------------------
    1. Columns whose NaN fraction exceeds ``_NAN_DROP_THRESHOLD`` (0.5, i.e.
       50 %) are dropped entirely.
    2. Remaining NaN values are imputed with the column mean (``np.nanmean``).
    3. Columns are centred by subtracting the (imputed) column mean.

    PCA is then performed via ``numpy.linalg.svd`` on the centred matrix.
    The number of components returned is capped at
    ``min(n_components, n_features_retained, n_snapshots - 1)``.
    Single-snapshot input (``n_snapshots < 2``) yields an empty
    :class:`PCAResult` as no variance can be estimated from one observation.

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

    # ---- Step 3: centre ----
    X_centered = X - X.mean(axis=0)

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
