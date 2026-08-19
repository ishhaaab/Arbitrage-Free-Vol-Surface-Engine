"""Tests for the fallback masking utility."""

import numpy as np

from arbfree_vol.plotting.masking import make_fallback_mask


def test_basic_masking():
    """Mask is True at indices corresponding to fallback T values."""
    grid_T = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    fallback_slices = [0.10, 0.25]
    mask = make_fallback_mask(grid_T, fallback_slices)
    expected = np.array([False, True, False, False, True, False])
    np.testing.assert_array_equal(mask, expected)


def test_empty_fallback_slices():
    """Empty fallback_slices returns all-False mask."""
    grid_T = np.array([0.05, 0.10, 0.15])
    fallback_slices: list[float] = []
    mask = make_fallback_mask(grid_T, fallback_slices)
    expected = np.array([False, False, False])
    np.testing.assert_array_equal(mask, expected)


def test_tolerance_approximate_match():
    """Mask handles approximate matches within the given tolerance."""
    grid_T = np.array([0.0849, 0.1000, 0.1753, 0.2575])
    # fallback_slices values are slightly off from grid_T
    fallback_slices = [0.0850, 0.1750, 0.2576]
    mask = make_fallback_mask(grid_T, fallback_slices, tol=0.01)
    # All grid_T should match within tol=0.01
    expected = np.array([True, False, True, True])
    np.testing.assert_array_equal(mask, expected)


def test_tolerance_tight():
    """With a very tight tolerance, only exact-ish matches pass."""
    grid_T = np.array([0.05, 0.1000, 0.1005, 0.20])
    fallback_slices = [0.1000]
    mask = make_fallback_mask(grid_T, fallback_slices, tol=0.0001)
    expected = np.array([False, True, False, False])
    np.testing.assert_array_equal(mask, expected)


def test_no_match_outside_tol():
    """Grid values far from any fallback T are all False."""
    grid_T = np.array([0.01, 0.02, 0.03])
    fallback_slices = [0.90, 1.00]
    mask = make_fallback_mask(grid_T, fallback_slices, tol=0.01)
    expected = np.array([False, False, False])
    np.testing.assert_array_equal(mask, expected)


def test_single_fallback_matches_multiple_grid_points():
    """A single fallback T can match multiple grid points if they
    are both within tolerance."""
    grid_T = np.array([0.099, 0.100, 0.101, 0.200])
    fallback_slices = [0.100]
    mask = make_fallback_mask(grid_T, fallback_slices, tol=0.002)
    expected = np.array([True, True, True, False])
    np.testing.assert_array_equal(mask, expected)


def test_empty_grid():
    """Empty grid returns empty mask."""
    grid_T = np.array([], dtype=np.float64)
    fallback_slices = [0.10]
    mask = make_fallback_mask(grid_T, fallback_slices)
    assert mask.shape == (0,)
    assert mask.dtype == bool
