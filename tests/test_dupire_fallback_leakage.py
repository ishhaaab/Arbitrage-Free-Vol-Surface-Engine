"""Tests for Dupire fallback-slice stencil contamination.

Constructs synthetic surfaces and verifies that NaN propagates correctly
from fallback slices into neighboring rows whose FD stencil crosses the
fallback boundary.
"""

import math

import pytest

from arbfree_vol.svi.model import SVIParams
from arbfree_vol.models.fitted import FittedSlice, FittedSurface
from arbfree_vol.pricing.local_vol import dupire


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _forward(T: float, spot: float = 100.0, r: float = 0.05,
             q: float = 0.0) -> float:
    return spot * math.exp((r - q) * T)


def _make_slice(T: float, sigma: float, spot: float = 100.0,
                r: float = 0.05, q: float = 0.0) -> FittedSlice:
    """Build a flat (b=0) FittedSlice at the given T and sigma."""
    return FittedSlice(
        expiry_time=T,
        params=SVIParams(
            a=sigma ** 2 * T, b=0.0, rho=0.0, m=0.0, sigma=0.2
        ),
        rmse=0.0,
        forward_price=_forward(T, spot, r, q),
        n_quotes_total=5,
        n_quotes_used=5,
    )


def _five_slice_surface(
    fallback_T: float = 0.5,
    fallback_sigma: float = 0.05,
    normal_sigma: float = 0.20,
) -> tuple[FittedSurface, list[float]]:
    """Build a 5-slice surface with T = [0.1, 0.3, 0.5, 0.7, 0.9].

    The fallback_T slice has ``fallback_sigma`` (deliberately low to
    break theta monotonicity), while all other slices use ``normal_sigma``.

    Returns the FittedSurface and the list of fallback T values.
    """
    spot, r, q = 100.0, 0.05, 0.0
    Ts = [0.1, 0.3, 0.5, 0.7, 0.9]
    slices = []
    fwd_curve = []
    for T in Ts:
        sigma = fallback_sigma if math.isclose(T, fallback_T) else normal_sigma
        slices.append(_make_slice(T, sigma, spot, r, q))
        fwd_curve.append((T, _forward(T, spot, r, q)))

    fs = FittedSurface(
        spot=spot,
        risk_free=r,
        div_yield=q,
        forward_curve=tuple(fwd_curve),
        fitted_slices=tuple(slices),
    )
    return fs, [fallback_T]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDupireFallbackLeakage:
    """Verify NaN propagation from fallback slices through FD stencil."""

    def test_fallback_row_is_nan(self) -> None:
        """The fallback slice itself (T=0.5) is NaN."""
        fs, fallback_Ts = _five_slice_surface()
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.1, 0.3, 0.5, 0.7, 0.9]

        lv = dupire(fs, strikes, maturities, fallback_slices=fallback_Ts)

        # T=0.5 is at index 2
        for iK in range(len(strikes)):
            assert math.isnan(lv.grid[2][iK]), (
                f"T=0.5, K={strikes[iK]}: expected NaN, got {lv.grid[2][iK]}"
            )

    def test_stencil_neighbor_T03_is_nan(self) -> None:
        """T=0.3 (stencil T+dT crosses into fallback interval) is NaN."""
        fs, fallback_Ts = _five_slice_surface()
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.1, 0.3, 0.5, 0.7, 0.9]

        lv = dupire(fs, strikes, maturities, fallback_slices=fallback_Ts)

        # T=0.3 is at index 1
        for iK in range(len(strikes)):
            assert math.isnan(lv.grid[1][iK]), (
                f"T=0.3, K={strikes[iK]}: expected NaN, got {lv.grid[1][iK]}"
            )

    def test_stencil_neighbor_T07_is_nan(self) -> None:
        """T=0.7 (stencil T-dT crosses into fallback interval) is NaN."""
        fs, fallback_Ts = _five_slice_surface()
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.1, 0.3, 0.5, 0.7, 0.9]

        lv = dupire(fs, strikes, maturities, fallback_slices=fallback_Ts)

        # T=0.7 is at index 3
        for iK in range(len(strikes)):
            assert math.isnan(lv.grid[3][iK]), (
                f"T=0.7, K={strikes[iK]}: expected NaN, got {lv.grid[3][iK]}"
            )

    def test_far_slice_T01_is_clean(self) -> None:
        """T=0.1 (stencil never reaches T=0.5) is clean finite."""
        fs, fallback_Ts = _five_slice_surface()
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.1, 0.3, 0.5, 0.7, 0.9]

        lv = dupire(fs, strikes, maturities, fallback_slices=fallback_Ts)

        # T=0.1 is at index 0
        for iK in range(len(strikes)):
            val = lv.grid[0][iK]
            assert not math.isnan(val), (
                f"T=0.1, K={strikes[iK]}: expected finite, got NaN"
            )

    def test_far_slice_T09_is_clean(self) -> None:
        """T=0.9 (stencil never reaches T=0.5) is clean finite."""
        fs, fallback_Ts = _five_slice_surface()
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.1, 0.3, 0.5, 0.7, 0.9]

        lv = dupire(fs, strikes, maturities, fallback_slices=fallback_Ts)

        # T=0.9 is at index 4
        for iK in range(len(strikes)):
            val = lv.grid[4][iK]
            assert not math.isnan(val), (
                f"T=0.9, K={strikes[iK]}: expected finite, got NaN"
            )

    def test_no_fallback_unchanged(self) -> None:
        """With fallback_slices=None on a clean surface, all rows are finite."""
        # Use a uniformly flat surface (no theta dip) so no calendar arb.
        spot, r, q = 100.0, 0.05, 0.0
        sigma = 0.20
        Ts = [0.1, 0.3, 0.5, 0.7, 0.9]
        slices = []
        fwd_curve = []
        for T in Ts:
            slices.append(_make_slice(T, sigma, spot, r, q))
            fwd_curve.append((T, _forward(T, spot, r, q)))

        fs = FittedSurface(
            spot=spot, risk_free=r, div_yield=q,
            forward_curve=tuple(fwd_curve),
            fitted_slices=tuple(slices),
        )

        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.1, 0.3, 0.5, 0.7, 0.9]

        lv = dupire(fs, strikes, maturities, fallback_slices=None)

        for iT in range(len(maturities)):
            for iK in range(len(strikes)):
                val = lv.grid[iT][iK]
                assert not math.isnan(val), (
                    f"NaN at T={maturities[iT]}, K={strikes[iK]} "
                    f"with fallback_slices=None"
                )

    def test_empty_fallback_unchanged(self) -> None:
        """With fallback_slices=[] on a clean surface, all rows are finite."""
        spot, r, q = 100.0, 0.05, 0.0
        sigma = 0.20
        Ts = [0.1, 0.3, 0.5, 0.7, 0.9]
        slices = []
        fwd_curve = []
        for T in Ts:
            slices.append(_make_slice(T, sigma, spot, r, q))
            fwd_curve.append((T, _forward(T, spot, r, q)))

        fs = FittedSurface(
            spot=spot, risk_free=r, div_yield=q,
            forward_curve=tuple(fwd_curve),
            fitted_slices=tuple(slices),
        )

        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.1, 0.3, 0.5, 0.7, 0.9]

        lv = dupire(fs, strikes, maturities, fallback_slices=[])

        for iT in range(len(maturities)):
            for iK in range(len(strikes)):
                val = lv.grid[iT][iK]
                assert not math.isnan(val), (
                    f"NaN at T={maturities[iT]}, K={strikes[iK]} "
                    f"with fallback_slices=[]"
                )

    def test_multiple_fallbacks(self) -> None:
        """Two fallback slices at T=0.3 and T=0.7 propagate NaN correctly."""
        spot, r, q = 100.0, 0.05, 0.0
        Ts = [0.1, 0.3, 0.5, 0.7, 0.9]
        fallback_Ts = [0.3, 0.7]
        slices = []
        fwd_curve = []
        for T in Ts:
            sigma = 0.05 if T in fallback_Ts else 0.20
            slices.append(_make_slice(T, sigma, spot, r, q))
            fwd_curve.append((T, _forward(T, spot, r, q)))

        fs = FittedSurface(
            spot=spot, risk_free=r, div_yield=q,
            forward_curve=tuple(fwd_curve),
            fitted_slices=tuple(slices),
        )

        strikes = [90.0, 100.0, 110.0]
        maturities = [0.1, 0.3, 0.5, 0.7, 0.9]
        lv = dupire(fs, strikes, maturities, fallback_slices=fallback_Ts)

        # T=0.1 (idx 0): stencil T+dT=0.101, bracketed by (0.1, 0.3).
        #   0.3 is fallback → NaN
        for iK in range(len(strikes)):
            assert math.isnan(lv.grid[0][iK]), (
                f"T=0.1, K={strikes[iK]}: expected NaN (neighbor 0.3 is fallback)"
            )

        # T=0.3 (idx 1): fallback itself → NaN
        for iK in range(len(strikes)):
            assert math.isnan(lv.grid[1][iK])

        # T=0.5 (idx 2): T-dT=0.499 in (0.3, 0.5) with 0.3 fallback;
        #   T+dT=0.501 in (0.5, 0.7) with 0.7 fallback → NaN
        for iK in range(len(strikes)):
            assert math.isnan(lv.grid[2][iK])

        # T=0.7 (idx 3): fallback itself → NaN
        for iK in range(len(strikes)):
            assert math.isnan(lv.grid[3][iK])

        # T=0.9 (idx 4): T-dT=0.899 in (0.7, 0.9) with 0.7 fallback → NaN
        for iK in range(len(strikes)):
            assert math.isnan(lv.grid[4][iK])

    def test_grid_shape_preserved(self) -> None:
        """Grid dimensions match inputs even with fallback masking."""
        fs, fallback_Ts = _five_slice_surface()
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.1, 0.3, 0.5, 0.7, 0.9]

        lv = dupire(fs, strikes, maturities, fallback_slices=fallback_Ts)

        assert len(lv.grid) == len(maturities)
        for row in lv.grid:
            assert len(row) == len(strikes)

    # ── Single-slice guard (regression for the one-slice IndexError) ──
    # Pre-fix, ``_stencil_touches_fallback`` computed ``fitted_times[idx + 1]``
    # on a single-element tuple: ``n - 2`` went negative and the clamped
    # index resolved to ``fitted_times[1]``, raising IndexError whenever a
    # grid row was not itself a fallback maturity.  dupire() only validated
    # >= 3 grid maturities, not >= 2 fitted slices, so a surface with ONE
    # fitted slice plus a non-empty fallback list crashed.

    def test_stencil_helper_single_fitted_slice_returns_false(self) -> None:
        """_stencil_touches_fallback with fewer than two fitted times must
        return False (no interior interval exists to contaminate) instead
        of IndexErroring on ``fitted_times[1]``."""
        from arbfree_vol.pricing._fallback import _stencil_touches_fallback

        # T=0.4 is not itself a fallback maturity, so the pre-fix code
        # reached the stencil-bracket loop with n=1 and crashed on
        # fitted_times[1].
        assert _stencil_touches_fallback(0.4, (0.5,), {0.5}, 1e-3) is False

    def test_single_fitted_slice_with_fallback_does_not_raise(self) -> None:
        """A FittedSurface with exactly ONE fitted slice plus a non-empty
        fallback_slices list must not raise IndexError, and the fallback
        maturity rows must still be NaN (grid tolerance behavior)."""
        fs, _ = _five_slice_surface()
        # Keep only the single fitted slice at T=0.5.
        single = FittedSurface(
            spot=fs.spot,
            risk_free=fs.risk_free,
            div_yield=fs.div_yield,
            forward_curve=fs.forward_curve,
            fitted_slices=(fs.fitted_slices[2],),
        )
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.5, 0.5, 0.5]

        # Must not raise IndexError.
        lv = dupire(single, strikes, maturities, fallback_slices=[0.5])

        # Every grid maturity is a fallback maturity, so every row is
        # masked NaN per the existing maturity-tolerance behavior.
        assert len(lv.grid) == len(maturities)
        for iT, row in enumerate(lv.grid):
            for iK in range(len(strikes)):
                assert math.isnan(row[iK]), (
                    f"T={maturities[iT]}, K={strikes[iK]}: expected NaN, "
                    f"got {row[iK]}"
                )

    def test_single_fitted_slice_non_fallback_row_raises_clear_error(self) -> None:
        """A one-fitted-slice surface requesting any non-fallback grid row
        must raise a CLEAR ValueError, not the obscure out-of-range error
        that used to leak from total_variance_at/_dw_dT."""
        fs, _ = _five_slice_surface()
        single = FittedSurface(
            spot=fs.spot,
            risk_free=fs.risk_free,
            div_yield=fs.div_yield,
            forward_curve=fs.forward_curve,
            fitted_slices=(fs.fitted_slices[2],),
        )
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]

        # T=0.5 is a fallback maturity; T=0.4 and T=0.6 are NOT, so the
        # surface would need to be evaluated and must fail clearly.
        maturities = [0.4, 0.5, 0.6]
        with pytest.raises(ValueError, match="dupire requires at least 2 fitted slices"):
            dupire(single, strikes, maturities, fallback_slices=[0.5])

        # Without a fallback list every row would be evaluated — same
        # clear error, no grid evaluation attempted.
        with pytest.raises(ValueError, match="dupire requires at least 2 fitted slices"):
            dupire(single, strikes, maturities)

    def test_plot_dupire_heatmap_uses_nan_masking(self) -> None:
        """plot_dupire_heatmap masks exactly the NaN cells of lv.grid.

        The plotted mesh array must match the NaN positions in the
        supplied grid cell-for-cell (same shape, same masked positions) —
        a fallback row that fails to mask, or an extra masked cell that
        is finite in the grid, fails the test.
        """
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")

        from arbfree_vol.viz.local_vol import plot_dupire_heatmap

        fs, fallback_Ts = _five_slice_surface()
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.1, 0.3, 0.5, 0.7, 0.9]

        lv = dupire(fs, strikes, maturities, fallback_slices=fallback_Ts)

        fig = plot_dupire_heatmap(lv, symbol="TEST", fallback_slices=fallback_Ts)
        mesh = fig.axes[0].collections[0]
        plotted = mesh.get_array()

        grid_np = np.array(lv.grid)
        assert plotted.shape == grid_np.shape, (
            f"plotted array shape {plotted.shape} != grid shape {grid_np.shape}"
        )
        expected_mask = np.isnan(grid_np)
        actual_mask = np.ma.getmaskarray(plotted)
        assert np.array_equal(actual_mask, expected_mask), (
            f"plotted mask must match the NaN cells of lv.grid exactly:\n"
            f"  grid NaN:\n{expected_mask}\n"
            f"  plotted mask:\n{actual_mask}"
        )

    def test_plot_dupire_heatmap_no_fallback_still_works(self) -> None:
        """plot_dupire_heatmap with no fallback must not mask any cell.

        Uses a uniformly flat surface (all slices at the same sigma) so
        no calendar-arb NaN leaks into the grid either: every cell is
        finite and the plotted mesh is fully unmasked.
        """
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")

        from arbfree_vol.viz.local_vol import plot_dupire_heatmap

        spot, r, q = 100.0, 0.05, 0.0
        sigma = 0.20
        Ts = [0.1, 0.3, 0.5, 0.7, 0.9]
        slices = []
        fwd_curve = []
        for T in Ts:
            slices.append(_make_slice(T, sigma, spot, r, q))
            fwd_curve.append((T, _forward(T, spot, r, q)))

        fs = FittedSurface(
            spot=spot, risk_free=r, div_yield=q,
            forward_curve=tuple(fwd_curve),
            fitted_slices=tuple(slices),
        )
        strikes = [90.0, 95.0, 100.0, 105.0, 110.0]
        maturities = [0.1, 0.3, 0.5, 0.7, 0.9]

        lv = dupire(fs, strikes, maturities)

        fig = plot_dupire_heatmap(lv, symbol="TEST")
        mesh = fig.axes[0].collections[0]
        plotted = mesh.get_array()

        assert plotted.shape == (len(maturities), len(strikes))
        assert not np.ma.getmaskarray(plotted).any(), (
            "no cells may be masked when no fallback exists and the "
            "surface is calendar-arb-free"
        )
