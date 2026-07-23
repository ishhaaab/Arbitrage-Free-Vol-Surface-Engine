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


def test_model_comparison_returns_figure() -> None:
    from arbfree_vol.viz.comparison import plot_model_comparison

    _, r = _two_expiry_surface()
    fig = plot_model_comparison({"SVI": r, "eSSVI": r})
    assert fig.axes is not None


def test_smile_model_comparison_returns_figure() -> None:
    from arbfree_vol.viz.smiles import plot_smile_model_comparison

    s, r = _two_expiry_surface()
    fig = plot_smile_model_comparison(s, {"SVI": r, "eSSVI": r})
    assert fig.axes is not None


def test_iv_heatmap_returns_figure() -> None:
    from arbfree_vol.viz.surface import plot_iv_heatmap
    from arbfree_vol.surface.interpolate import build_fitted_surface

    _, r = _two_expiry_surface()
    fs = build_fitted_surface(r)
    fig = plot_iv_heatmap(fs)
    assert fig.axes is not None


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
    assert fig.axes is not None


def test_greeks_heatmap_returns_figure() -> None:
    from arbfree_vol.viz.risk import plot_greeks_heatmap
    from arbfree_vol.surface.interpolate import build_fitted_surface

    _, r = _two_expiry_surface()
    fs = build_fitted_surface(r)
    fig = plot_greeks_heatmap(fs, [90, 100, 110], [0.5, 1.0])
    assert fig.axes is not None


def test_scenario_payoff_returns_figure() -> None:
    from arbfree_vol.viz.risk import plot_scenario_payoff
    from arbfree_vol.surface.interpolate import build_fitted_surface
    from arbfree_vol.surface.risk import spot_bump_analysis
    from arbfree_vol.models.option import OptionContract

    _, r = _two_expiry_surface()
    fs = build_fitted_surface(r)
    T = fs.fitted_slices[-1].expiry_time
    spot = fs.spot
    positions = [
        (OptionContract(symbol="X", option_type=OptionType.CALL,
                        strike=round(spot), expiry_date=_DUMMY),
         T, 1.0),
    ]
    scenarios = spot_bump_analysis(fs, positions, bumps=[-0.05, 0.0, 0.05])
    fig = plot_scenario_payoff(scenarios)
    assert fig.axes is not None
