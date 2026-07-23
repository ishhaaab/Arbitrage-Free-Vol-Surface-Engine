# arbfree-vol-surface

Arbitrage-free implied volatility surface engine. Calibrates SVI, eSSVI, and SABR
total-variance parameterizations to cleaned option market quotes, detects static
arbitrage violations, repairs noisy surfaces, and provides a full surface-query
layer (IV interpolation, portfolio Greeks, scenario risk, Dupire local vol, PCA
deformations). Built for correctness, numerical stability, and clean architecture.

## Quick start

```bash
pytest tests/ -q                              # 186 tests
python demo/full_pipeline_demo.py             # end-to-end demo (no network)
python examples/yfinance_demo.py              # live SPY demo (needs network)
python examples/backtest_demo.py              # live SPY mispricing backtest
```

## Live Showcase (SPY options)

Run `python examples/yfinance_demo.py` to fetch live SPY options, repair with all
three smile models (SVI / eSSVI / SABR), interpolate the fitted surface, extract
Dupire local volatility, compute portfolio Greeks, run spot-bump scenarios, and
produce 8 plots.

### 3D implied volatility surface

![SPY 3D surface](examples/yfinance_demo_surface.png)

### Smile model comparison (SVI / eSSVI / SABR)

![Smile model comparison](examples/yfinance_demo_smiles_comparison.png)

### Model fit quality comparison

![Model fit comparison](examples/yfinance_demo_model_comparison.png)

### Implied vol heatmap (via iv_at)

![IV heatmap](examples/yfinance_demo_iv_heatmap.png)

### Dupire local volatility

![Dupire local vol](examples/yfinance_demo_dupire.png)

### Portfolio Greeks (delta / gamma / vega)

![Greeks heatmap](examples/yfinance_demo_greeks.png)

### Spot-bump scenario P&L

![Scenario P&L](examples/yfinance_demo_scenario.png)

### Repair comparison (before vs after)

![Repair comparison](examples/yfinance_demo_repair.png)

## What's built

| Layer | Module | Capabilities |
|---|---|---|
| Pricing | `pricing/` | Black-Scholes, analytic Greeks (δ/γ/ν/θ/ρ), IV solver (Newton + Brent) |
| Models | `models/` | `OptionContract`, `Quote`, `ExpirySlice`, `VolSurface` (Pydantic boundary types) |
| Ingestion | `ingestion/` | Yahoo Finance fetcher, CSV loader, 8 cleaning rules with audit records |
| Arbitrage | `arbitrage/` | Quote-level (parity, monotonicity, butterfly, calendar, wide-spread); curve-level (min-variance, `g(k)` butterfly, calendar) |
| Smile calibration | `svi/` `ssvi/` `sabr/` | Raw SVI (5-param), eSSVI (Gatheral-Jacquier), SABR (Hagan 2002 asymptotic). **All** map to raw SVI for downstream compatibility. Constrained SVI calibration penalises butterfly-arb during fitting. |
| Repair | `repair/` | Reject violating quotes, estimate forward curve (median), refit slices, re-detect. Pluggable: `repair(use_ssvi=True)` or `repair(use_sabr=True)`. |
| Surface interpolation | `surface/` | `FittedSurface`, `iv_at(K,T)` (linear-in-T total-variance, per-slice forwards). `PortfolioGreeks`, `bucketed_greeks`. `spot_bump_analysis`, `vol_bump_analysis`, `parallel_vega_pnl`. |
| Dupire local vol | `pricing/local_vol.py` | Gatheral SSVI-compatible strip-out via finite differences, flat-vol benchmark, calendar-arb guard. |
| Surface dynamics | `dynamics.py` | `SurfaceSeries`, SVD-based PCA on parameter matrix, dominant deformation modes (Level/Tilt/Curvature). |
| Backtest | `backtest/` | Cross-sectional mispricing backtest: detect `|market_IV − model_IV| > threshold`, delta-hedged trades (frozen-vol), Sharpe / hit-rate / drawdown / P&L percentiles |

## Running the demo

All modules exercise on synthetic data with no network dependency:

```bash
python demo/full_pipeline_demo.py
```

This runs: repair comparison (SVI / eSSVI / SABR) → `iv_at` query grid → portfolio Greeks + spot bumps → Dupire flat-vol benchmark → PCA on a synthetic time series. Full output in ~2 seconds.

## Key design conventions

- **Boundary vs compute**: Pydantic `BaseModel` at I/O boundaries; frozen `@dataclass(slots=True)` everywhere else (Greeks, violations, fitted slices, scenarios, PCA output).
- **Report, not raises**: Detection enumerates all violations in one pass.
- **Median, not mean**: Forward curve uses median across parity pairs.
- **Pluggable smile models**: `repair(use_ssvi=False, use_sabr=False)` with mutual exclusivity. `to_raw_svi_params()` adapters keep downstream pipeline unchanged.
- **Constrained calibration**: `calibrate_constrained(points, arb_penalty=100.0)` augments the residual with a butterfly-arb penalty `√penalty·√max(−g(k),0)` on a k-grid.

## Roadmap

See `docs/roadmap.md`. Milestones 3 (PCA dynamics), 5 (SABR), 7 (Dupire),
and 9 (Backtest — Design B) are complete. Planned next: rolling daily-refit
backtest (Milestone 12).

## Tech stack

Python 3.12+, NumPy, SciPy, Pydantic, Pytest.

No sklearn — PCA uses `numpy.linalg.svd`. No FastAPI/Streamlit yet.
