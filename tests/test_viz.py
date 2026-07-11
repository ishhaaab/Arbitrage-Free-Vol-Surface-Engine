"""Smoke tests for the visualization module — just check they return figures."""

from datetime import date

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


def test_surface_plot_returns_figure() -> None:
    from arbfree_vol.viz.surface import plot_surface

    _, r = _two_expiry_surface()
    fig = plot_surface(list(r.fitted_slices))
    assert fig.axes is not None


def test_smiles_plot_returns_figure() -> None:
    from arbfree_vol.viz.smiles import plot_smiles

    s, r = _two_expiry_surface()
    fig = plot_smiles(s, list(r.fitted_slices))
    assert fig.axes is not None


def test_violations_plot_returns_figure() -> None:
    from arbfree_vol.viz.violations import plot_violations_bar

    s, r = _two_expiry_surface()
    v_report = detect(s)
    fig = plot_violations_bar(v_report)
    assert fig.axes is not None


def test_comparison_plot_returns_figure() -> None:
    from arbfree_vol.viz.comparison import plot_comparison

    _, r = _two_expiry_surface()
    fig = plot_comparison(r, r)
    assert fig.axes is not None


def test_heatmap_2d_returns_figure() -> None:
    from arbfree_vol.viz.surface import plot_heatmap_2d

    _, r = _two_expiry_surface()
    fig = plot_heatmap_2d(list(r.fitted_slices))
    assert fig.axes is not None


def test_smiles_heatmap_returns_figure() -> None:
    from arbfree_vol.viz.smiles import plot_smiles_heatmap

    _, r = _two_expiry_surface()
    fig = plot_smiles_heatmap(list(r.fitted_slices))
    assert fig.axes is not None
