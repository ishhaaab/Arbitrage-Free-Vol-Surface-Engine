"""Smoke tests for the visualization module — just check they return figures."""

from datetime import date

import matplotlib
matplotlib.use("Agg")

import numpy as np

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


# ---------------------------------------------------------------------------
# Backtest viz smoke tests
# ---------------------------------------------------------------------------
def _fake_backtest_result(n: int = 4) -> object:
    """Build a synthetic ``BacktestResult`` with ``n`` trades (both sides)."""
    from datetime import timedelta
    from arbfree_vol.models.option import OptionType
    from arbfree_vol.backtest.types import (
        BacktestResult,
        MispricingSignal,
        Trade,
        TradePnL,
    )

    base = date(2030, 6, 1)
    signals: list[MispricingSignal] = []
    trades: list[Trade] = []
    pnls: list[TradePnL] = []

    for i in range(n):
        side = 1 if i % 2 == 0 else -1
        expiry = base + timedelta(days=30 + i * 10)
        sig = MispricingSignal(
            strike=100.0 + i * 5.0,
            expiry_time=0.08 + i * 0.03,
            expiry_date=expiry,
            option_type=OptionType.CALL if side == 1 else OptionType.PUT,
            market_iv=0.20 + 0.02 * i,
            model_iv=0.20,
            mispricing=0.02 * i if side == 1 else -0.02 * i,
            entry_price=2.0 + i * 0.5,
            side=side,
        )
        signals.append(sig)
        tr = Trade(
            signal=sig,
            entry_date=base,
            entry_spot=100.0,
            frozen_vol=sig.market_iv,
            quantity=side,
            risk_free=0.05,
            div_yield=0.0,
        )
        trades.append(tr)
        pnl_val = (0.5 + i * 0.5) * side  # varied P&L
        pnls.append(
            TradePnL(
                trade=tr,
                realized_pnl=pnl_val,
                option_pnl=pnl_val * 0.7,
                hedge_pnl=pnl_val * 0.3,
                expiry_spot=102.0 + i,
                hold_days=30,
                hit=pnl_val > 0.0,
            )
        )

    realized = np.array([p.realized_pnl for p in pnls], dtype=float)
    n_t = len(trades)
    hit_rate = sum(1 for p in pnls if p.hit) / n_t if n_t > 0 else 0.0
    total = float(np.sum(realized))
    mean = total / n_t if n_t > 0 else 0.0
    std = float(np.std(realized, ddof=1)) if n_t > 1 else 0.0
    sharpe = mean / std if std > 0.0 and n_t >= 2 else 0.0
    sorted_pnls = sorted(pnls, key=lambda p: p.trade.signal.expiry_date)
    cumulative = np.cumsum(np.array([p.realized_pnl for p in sorted_pnls]))
    if len(cumulative) > 0:
        running_max = np.maximum.accumulate(cumulative)
        max_dd = float(np.max(running_max - cumulative))
    else:
        max_dd = 0.0
    p5 = float(np.percentile(realized, 5)) if n_t >= 1 else 0.0
    p50 = float(np.percentile(realized, 50)) if n_t >= 1 else 0.0
    p95 = float(np.percentile(realized, 95)) if n_t >= 1 else 0.0

    return BacktestResult(
        trades=tuple(trades),
        pnls=tuple(pnls),
        n_trades=n_t,
        hit_rate=hit_rate,
        total_pnl=total,
        mean_pnl=mean,
        std_pnl=std,
        sharpe=sharpe,
        max_drawdown=max_dd,
        pnl_p5=p5,
        pnl_p50=p50,
        pnl_p95=p95,
    )


def test_pnl_distribution_returns_figure() -> None:
    from arbfree_vol.viz.backtest import plot_pnl_distribution

    result = _fake_backtest_result()
    fig = plot_pnl_distribution(result, symbol="TEST")
    assert fig.axes is not None


def test_cumulative_pnl_returns_figure() -> None:
    from arbfree_vol.viz.backtest import plot_cumulative_pnl

    result = _fake_backtest_result()
    fig = plot_cumulative_pnl(result, symbol="TEST")
    assert fig.axes is not None


def test_mispricing_vs_pnl_returns_figure() -> None:
    from arbfree_vol.viz.backtest import plot_mispricing_vs_pnl

    result = _fake_backtest_result()
    fig = plot_mispricing_vs_pnl(result, symbol="TEST")
    assert fig.axes is not None


def test_backtest_metrics_returns_figure() -> None:
    from arbfree_vol.viz.backtest import plot_backtest_metrics

    result = _fake_backtest_result()
    fig = plot_backtest_metrics(result, symbol="TEST")
    assert fig.axes is not None


def test_backtest_empty_result_handling() -> None:
    """All four backtest viz functions handle n_trades == 0 gracefully."""
    from arbfree_vol.viz.backtest import (
        plot_pnl_distribution,
        plot_cumulative_pnl,
        plot_mispricing_vs_pnl,
        plot_backtest_metrics,
    )
    from arbfree_vol.backtest.types import BacktestResult

    empty = BacktestResult(
        trades=(),
        pnls=(),
        n_trades=0,
        hit_rate=0.0,
        total_pnl=0.0,
        mean_pnl=0.0,
        std_pnl=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
        pnl_p5=0.0,
        pnl_p50=0.0,
        pnl_p95=0.0,
    )

    fig1 = plot_pnl_distribution(empty)
    assert fig1.axes is not None

    fig2 = plot_cumulative_pnl(empty)
    assert fig2.axes is not None

    fig3 = plot_mispricing_vs_pnl(empty)
    assert fig3.axes is not None

    fig4 = plot_backtest_metrics(empty)
    assert fig4.axes is not None
