"""Smoke tests for the visualization module.

Beyond returning figures, these pin the plotted CONTENT: heatmaps must
carry mesh arrays of the expected shape (and mask fallback rows when
supplied), and line plots must carry non-empty line data with the
expected title / legend labels.  Agg backend is pinned for determinism.
"""

from datetime import date

import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.repair.engine import repair
from arbfree_vol.arbitrage.quote_detect import detect


_DUMMY = date(2030, 1, 1)
SPOT = 100.0
R = 0.05


def _bp(otype, K, sigma=0.2, tt=1.0):
    from arbfree_vol.models.option import OptionContract, BlackScholesInput
    from arbfree_vol.pricing.black_scholes import price
    c = OptionContract(symbol="X", option_type=otype, strike=K, expiry_date=_DUMMY)
    m = BlackScholesInput(contract=c, spot=SPOT, expiry_time=tt,
                          risk_free=R, div_yield=0.0, volatility=sigma)
    return price(m)


def _two_expiry_surface() -> tuple[VolSurface, object]:
    strikes = [80, 90, 100, 110, 120]
    qs1 = [Quote(strike=K, option_type=o, price=_bp(o, K, tt=0.5))
           for K in strikes for o in [OptionType.CALL, OptionType.PUT]]
    qs2 = [Quote(strike=K, option_type=o, price=_bp(o, K, tt=1.0))
           for K in strikes for o in [OptionType.CALL, OptionType.PUT]]
    s = VolSurface(spot=SPOT, risk_free=R, div_yield=0.0, slices=[
        ExpirySlice(expiry_time=0.5, quotes=qs1),
        ExpirySlice(expiry_time=1.0, quotes=qs2),
    ])
    r = repair(s)
    return s, r


def _legend_labels(ax) -> list[str]:
    legend = ax.get_legend()
    if legend is None:
        return []
    return [t.get_text() for t in legend.get_texts()]


def test_surface_plot_returns_figure() -> None:
    from arbfree_vol.viz.surface import plot_surface

    _, r = _two_expiry_surface()
    fig = plot_surface(list(r.fitted_slices))
    ax = fig.axes[0]

    # One SVI ribbon line per fitted expiry, each with non-empty data.
    assert len(ax.lines) == len(r.fitted_slices) == 2
    for line in ax.lines:
        xs, _, _ = line.get_data_3d()
        assert len(xs) > 0

    assert ax.get_title() == "Fitted SVI per-expiry smiles (2 expiries)"


def test_smiles_plot_returns_figure() -> None:
    from arbfree_vol.viz.smiles import plot_smiles

    s, r = _two_expiry_surface()
    fig = plot_smiles(s, list(r.fitted_slices))

    # One subplot per expiry, each with a non-empty fitted curve.
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert len(ax.lines) == 1
        assert len(ax.lines[0].get_xdata()) > 0
        assert ax.get_title() != ""
        assert "SVI fit" in _legend_labels(ax)


def test_violations_plot_returns_figure() -> None:
    from arbfree_vol.viz.violations import plot_violations_bar

    s, r = _two_expiry_surface()
    v_report = detect(s)
    fig = plot_violations_bar(v_report)
    assert fig.axes is not None
    # The bar chart axes must be configured with the violation title.
    assert "Arbitrage violations" in fig.axes[0].get_title()


def test_comparison_plot_returns_figure() -> None:
    from arbfree_vol.viz.comparison import plot_comparison

    _, r = _two_expiry_surface()
    fig = plot_comparison(r, r)
    assert fig.axes is not None
    # Three subplots, each with bar patches.
    assert len(fig.axes) == 3
    assert all(len(ax.patches) > 0 for ax in fig.axes)


def test_heatmap_2d_returns_figure() -> None:
    from arbfree_vol.viz.surface import plot_heatmap_2d

    _, r = _two_expiry_surface()
    fig = plot_heatmap_2d(list(r.fitted_slices))
    mesh = fig.axes[0].collections[0]
    arr = mesh.get_array()

    # Grid is n_k x n_T = (200, 150) after transposition.
    assert arr.shape == (200, 150)
    # At least some cells must be valid (outside-hull cells are masked).
    assert np.ma.getmaskarray(arr).sum() < arr.size


def test_smiles_heatmap_returns_figure() -> None:
    from arbfree_vol.viz.smiles import plot_smiles_heatmap

    _, r = _two_expiry_surface()
    fig = plot_smiles_heatmap(list(r.fitted_slices))
    mesh = fig.axes[0].collections[0]
    arr = mesh.get_array()

    # One row per fitted expiry, n_k columns.
    assert arr.shape == (2, 150)
    assert np.ma.getmaskarray(arr).sum() < arr.size


def test_model_comparison_returns_figure() -> None:
    from arbfree_vol.viz.comparison import plot_model_comparison

    _, r = _two_expiry_surface()
    fig = plot_model_comparison({"SVI": r, "eSSVI": r})
    assert fig.axes is not None
    # Two subplots, each with a bar per model.
    assert len(fig.axes) == 2
    assert all(len(ax.patches) == 2 for ax in fig.axes)


def test_smile_model_comparison_returns_figure() -> None:
    from arbfree_vol.viz.smiles import plot_smile_model_comparison

    s, r = _two_expiry_surface()
    fig = plot_smile_model_comparison(s, {"SVI": r, "eSSVI": r})

    # One subplot per expiry; each plots one line per model plus data.
    assert len(fig.axes) == 2
    for ax in fig.axes:
        assert len(ax.lines) == 2
        assert all(len(line.get_xdata()) > 0 for line in ax.lines)
        labels = _legend_labels(ax)
        assert "SVI" in labels and "eSSVI" in labels
        assert ax.get_title() != ""

    assert fig._suptitle is not None
    assert fig._suptitle.get_text() == "SPY smile model comparison"


def test_iv_heatmap_returns_figure() -> None:
    from arbfree_vol.viz.surface import plot_iv_heatmap
    from arbfree_vol.surface.interpolate import build_fitted_surface

    _, r = _two_expiry_surface()
    fs = build_fitted_surface(r)

    # Clean surface (no fallback): (n_maturities, n_strikes) = (50, 50),
    # fully finite and unmasked.
    fig = plot_iv_heatmap(fs)
    mesh = fig.axes[0].collections[0]
    arr = mesh.get_array()
    assert arr.shape == (50, 50)
    assert not np.ma.getmaskarray(arr).any()
    assert np.isfinite(arr.data).all()

    # Supplying a fallback maturity masks exactly the grid row closest
    # to it (T=0.5 is the first row of linspace(0.5, 1.0, 50)).
    fig_fb = plot_iv_heatmap(fs, fallback_slices=[0.5])
    arr_fb = fig_fb.axes[0].collections[0].get_array()
    assert arr_fb.shape == (50, 50)
    mask_rows = np.ma.getmaskarray(arr_fb).any(axis=1)
    assert mask_rows.sum() == 1
    assert mask_rows[0], "the T=0.5 grid row must be masked"


def test_dupire_heatmap_returns_figure() -> None:
    from arbfree_vol.viz.local_vol import plot_dupire_heatmap
    from arbfree_vol.pricing.local_vol import LocalVolSurface

    lv = LocalVolSurface(
        strikes=(90, 95, 100, 105, 110),
        maturities=(0.5, 1.0),
        grid=((0.2, 0.2, 0.2, 0.2, 0.2),
              (0.2, 0.2, 0.2, 0.2, 0.2)),
    )
    fig = plot_dupire_heatmap(lv)
    mesh = fig.axes[0].collections[0]
    arr = mesh.get_array()

    # (n_maturities, n_strikes) = (2, 5), fully finite on a clean grid.
    assert arr.shape == (2, 5)
    assert not np.ma.getmaskarray(arr).any()
    assert np.isfinite(arr.data).all()


def test_greeks_heatmap_returns_figure() -> None:
    from arbfree_vol.viz.risk import plot_greeks_heatmap
    from arbfree_vol.surface.interpolate import build_fitted_surface

    _, r = _two_expiry_surface()
    fs = build_fitted_surface(r)
    fig = plot_greeks_heatmap(fs, [90, 100, 110], [0.5, 1.0])

    # One subplot per Greek (default delta/gamma/vega); the colorbar
    # axes interleave with the subplot axes in fig.axes.
    subplot_axes = [
        ax for ax in fig.axes
        if ax.get_title() in {"Delta", "Gamma", "Vega"}
    ]
    assert len(subplot_axes) == 3
    for ax in subplot_axes:
        mesh = ax.collections[0]
        arr = mesh.get_array()
        # (n_maturities, n_strikes) = (2, 3), fully finite.
        assert arr.shape == (2, 3)
        assert not np.ma.getmaskarray(arr).any()
        assert np.isfinite(arr.data).all()


def test_greeks_heatmap_fallback_masking_content() -> None:
    """The fallback masking must match the expected masked cells EXACTLY.

    With fallback_slices=[0.5] on a [0.5, 1.0] maturity grid,
    make_fallback_mask marks row 0 (T=0.5) and row 0 only; every Greek
    heatmap must carry that mask cell-for-cell and keep the unmasked
    content equal to the bucketed Greek grid.
    """
    from arbfree_vol.viz.risk import plot_greeks_heatmap
    from arbfree_vol.surface.interpolate import build_fitted_surface
    from arbfree_vol.surface.greeks import bucketed_greeks

    _, r = _two_expiry_surface()
    fs = build_fitted_surface(r)
    strikes = [90, 100, 110]
    maturities = [0.5, 1.0]

    fig = plot_greeks_heatmap(fs, strikes, maturities, fallback_slices=[0.5])
    subplot_axes = [
        ax for ax in fig.axes
        if ax.get_title() in {"Delta", "Gamma", "Vega"}
    ]
    assert len(subplot_axes) == 3

    expected_mask = np.zeros((2, 3), dtype=bool)
    expected_mask[0, :] = True  # T=0.5 row is the fallback row

    greeks = bucketed_greeks(
        fs, strikes, maturities, OptionType.CALL,
        r=fs.risk_free, q=fs.div_yield,
    )
    for ax in subplot_axes:
        name = ax.get_title().lower()
        mesh = ax.collections[0]
        arr = mesh.get_array()
        assert arr.shape == (2, 3)
        actual_mask = np.ma.getmaskarray(arr)
        assert np.array_equal(actual_mask, expected_mask), (
            f"{name}: masked cells must equal the fallback maturity row "
            f"exactly:\nexpected:\n{expected_mask}\ngot:\n{actual_mask}"
        )
        # Unmasked content must be the untouched bucketed Greek grid.
        assert np.array_equal(arr.data[~expected_mask], greeks[name].T[~expected_mask]), (
            f"{name}: unmasked cells must match bucketed_greeks output"
        )


def test_greeks_heatmap_no_fallback_masks_nothing() -> None:
    """Without fallback slices on a clean surface, no heatmap cell may be
    masked."""
    from arbfree_vol.viz.risk import plot_greeks_heatmap
    from arbfree_vol.surface.interpolate import build_fitted_surface

    _, r = _two_expiry_surface()
    fs = build_fitted_surface(r)

    fig = plot_greeks_heatmap(fs, [90, 100, 110], [0.5, 1.0])
    subplot_axes = [
        ax for ax in fig.axes
        if ax.get_title() in {"Delta", "Gamma", "Vega"}
    ]
    assert len(subplot_axes) == 3
    for ax in subplot_axes:
        arr = ax.collections[0].get_array()
        assert not np.ma.getmaskarray(arr).any(), (
            "no cells may be masked when no fallback slices are supplied"
        )
        assert np.isfinite(arr.data).all()


def test_masked_heatmaps_use_configured_bad_color() -> None:
    """Regression: ``cmap.with_extremes(bad=...)`` returns a NEW colormap.

    The heatmap helpers must assign the result back to ``cmap``; if they
    discard it (the pre-fix behavior), masked fallback / NaN cells keep
    matplotlib's default fully-transparent bad color instead of the
    configured gray.  After plotting masked data, the effective colormap
    on the rendered mesh must carry the configured gray bad color.
    """
    from arbfree_vol.viz.surface import plot_iv_heatmap
    from arbfree_vol.viz.risk import plot_greeks_heatmap
    from arbfree_vol.viz.local_vol import plot_dupire_heatmap
    from arbfree_vol.surface.interpolate import build_fitted_surface
    from arbfree_vol.pricing.local_vol import LocalVolSurface

    expected_bad = (0.5019607843137255, 0.5019607843137255,
                    0.5019607843137255, 0.5)
    default_bad = (0.0, 0.0, 0.0, 0.0)

    _, r = _two_expiry_surface()
    fs = build_fitted_surface(r)

    # plot_iv_heatmap: fallback slice masks the T=0.5 maturity row.
    fig = plot_iv_heatmap(fs, fallback_slices=[0.5])
    mesh = fig.axes[0].collections[0]
    assert np.ma.getmaskarray(mesh.get_array()).any(), (
        "precondition: iv heatmap must have masked cells"
    )
    bad = mesh.get_cmap().get_bad()
    assert np.allclose(bad, expected_bad), (
        f"iv heatmap bad color must be the configured gray {expected_bad}, "
        f"got {bad}"
    )
    assert not np.allclose(bad, default_bad), (
        "iv heatmap bad color must not be the default transparent"
    )

    # plot_greeks_heatmap: fallback slice masks the T=0.5 maturity row
    # in every Greek subplot.
    fig = plot_greeks_heatmap(fs, [90, 100, 110], [0.5, 1.0],
                              fallback_slices=[0.5])
    subplot_axes = [
        ax for ax in fig.axes
        if ax.get_title() in {"Delta", "Gamma", "Vega"}
    ]
    assert len(subplot_axes) == 3
    for ax in subplot_axes:
        mesh = ax.collections[0]
        assert np.ma.getmaskarray(mesh.get_array()).any(), (
            "precondition: greek heatmap must have masked cells"
        )
        bad = mesh.get_cmap().get_bad()
        assert np.allclose(bad, expected_bad), (
            f"greek heatmap bad color must be the configured gray "
            f"{expected_bad}, got {bad}"
        )

    # plot_dupire_heatmap: a NaN cell in the local-vol grid is masked.
    lv = LocalVolSurface(
        strikes=(90, 95, 100, 105, 110),
        maturities=(0.5, 1.0),
        grid=((0.2, 0.2, 0.2, 0.2, 0.2),
              (0.2, float("nan"), 0.2, 0.2, 0.2)),
    )
    fig = plot_dupire_heatmap(lv)
    mesh = fig.axes[0].collections[0]
    assert np.ma.getmaskarray(mesh.get_array()).any(), (
        "precondition: dupire heatmap must have masked cells"
    )
    bad = mesh.get_cmap().get_bad()
    assert np.allclose(bad, expected_bad), (
        f"dupire heatmap bad color must be the configured gray "
        f"{expected_bad}, got {bad}"
    )


# ---------------------------------------------------------------------------
# Edge-case guards in viz/surface.py
# ---------------------------------------------------------------------------

def test_plot_surface_raises_with_less_than_two_slices() -> None:
    from arbfree_vol.viz.surface import plot_surface
    from arbfree_vol.models.fitted import FittedSlice
    from arbfree_vol.svi.model import SVIParams

    single = FittedSlice(
        expiry_time=1.0,
        params=SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.2),
        rmse=0.0,
        forward_price=100.0,
        n_quotes_total=5,
        n_quotes_used=5,
    )

    with pytest.raises(ValueError, match="at least two fitted slices"):
        plot_surface([single])

    with pytest.raises(ValueError, match="at least two fitted slices"):
        plot_surface([])


def test_plot_heatmap_2d_raises_with_less_than_two_slices() -> None:
    from arbfree_vol.viz.surface import plot_heatmap_2d
    from arbfree_vol.models.fitted import FittedSlice
    from arbfree_vol.svi.model import SVIParams

    single = FittedSlice(
        expiry_time=1.0,
        params=SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.2),
        rmse=0.0,
        forward_price=100.0,
        n_quotes_total=5,
        n_quotes_used=5,
    )

    with pytest.raises(ValueError, match="at least two fitted slices"):
        plot_heatmap_2d([single])


def test_plot_heatmap_2d_raises_with_too_few_points() -> None:
    """Fewer than 5 data points across all slices -> ValueError."""
    from arbfree_vol.viz.surface import plot_heatmap_2d
    from arbfree_vol.models.fitted import FittedSlice
    from arbfree_vol.svi.model import SVIParams

    sl1 = FittedSlice(
        expiry_time=0.5,
        params=SVIParams(a=0.02, b=0.0, rho=0.0, m=0.0, sigma=0.2),
        rmse=0.0,
        forward_price=100.0,
        n_quotes_total=5,
        n_quotes_used=5,
        data_points=((0.0, 0.02), (0.1, 0.025)),
    )
    sl2 = FittedSlice(
        expiry_time=1.0,
        params=SVIParams(a=0.04, b=0.0, rho=0.0, m=0.0, sigma=0.2),
        rmse=0.0,
        forward_price=100.0,
        n_quotes_total=5,
        n_quotes_used=5,
        data_points=((0.0, 0.04), (0.1, 0.045)),
    )

    with pytest.raises(ValueError, match="Not enough data points"):
        plot_heatmap_2d([sl1, sl2])


def test_plot_iv_heatmap_raises_with_no_slices() -> None:
    from arbfree_vol.viz.surface import plot_iv_heatmap
    from arbfree_vol.models.fitted import FittedSurface

    fs = FittedSurface(
        spot=100.0,
        risk_free=0.05,
        div_yield=0.0,
        forward_curve=(),
        fitted_slices=(),
    )

    with pytest.raises(ValueError, match="no slices"):
        plot_iv_heatmap(fs)


def test_plot_iv_heatmap_masks_out_of_range_cells() -> None:
    """iv_at raising ValueError for out-of-range strikes/expiries is
    absorbed: those grid cells stay NaN (masked) instead of aborting."""
    from arbfree_vol.viz.surface import plot_iv_heatmap
    from arbfree_vol.surface.interpolate import build_fitted_surface

    # A surface whose slices have no fitted slices beyond their own range:
    # iv_at over a wider grid will raise ValueError on out-of-range cells.
    _, r = _two_expiry_surface()
    fs = build_fitted_surface(r)

    # Query a grid wider than the surface's own strike range by passing
    # a spot-relative strike grid that extends outside the fitted surface.
    fig = plot_iv_heatmap(fs)
    mesh = fig.axes[0].collections[0]
    arr = mesh.get_array()
    assert arr.shape == (50, 50)
    # Some cells may be masked if any iv_at call raised.
    assert np.ma.getmaskarray(arr).shape == (50, 50)
