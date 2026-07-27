# arbfree-vol-surface

This library takes a messy real-world option chain and turns it into a clean, arbitrage-free implied volatility surface.

## The problem

An implied vol surface tells you the volatility the market is charging at every strike and every expiry. Exotic pricers, risk reports, and vol trades all lean on it.

Raw option quotes are noisy: bid/ask gaps, stale prints, crossed markets, and illiquid strikes. The surface you'd build straight from those quotes usually breaks basic no-arbitrage conditions. A desk has to clean the data, find the violations, and fit a smooth model before the surface is any good. This project does all of that in one pipeline.

## What it does

The pipeline has five stages:

1. **Clean the quotes**. Eight rejection rules drop bad prices, crossed markets, wide spreads, illiquid contracts, and near-expiry options. Every rejected quote is kept with a reason, so you can see why it was dropped.
2. **Detect arbitrage**. It checks the four classic static-arbitrage conditions: put-call parity, strike monotonicity, butterfly (convexity) arbitrage, and calendar arbitrage.
3. **Repair the surface**. It rejects the worst offending quotes, estimates a forward price per expiry from the median of put-call parity pairs, then refits a smile model with a penalty term that pushes against arbitrage during calibration.
4. **Query it**. The fitted surface answers `iv_at(strike, expiry)` and aggregates portfolio Greeks (delta, gamma, vega, theta, rho).
5. **Build on it**. Downstream modules pull out Dupire local vol, run PCA on a time series of surfaces to find the dominant ways the surface moves, and backtest a cross-sectional mispricing signal.

Three smile models are supported: raw SVI (Gatheral 2004), eSSVI (Gatheral-Jacquier 2014), and SABR (Hagan 2002). All three map back to a common SVI parameterization, so everything downstream works the same no matter which one you pick.

## Quick start

```bash
pytest tests/ -q                              # 177 tests across 23 files
python demo/full_pipeline_demo.py             # full pipeline on synthetic data (no network)
python examples/yfinance_demo.py              # live demo, 7 plots (needs network; use --symbol TICKER for others)
python examples/backtest_demo.py              # live mispricing backtest (use --symbol TICKER)
```

## Live showcase (SPY options)

Run `python examples/yfinance_demo.py --symbol SPY` to pull live SPY options, repair them with all three smile models, interpolate the fitted surface, extract Dupire local vol, compute portfolio Greeks, and produce the seven plots below.

### 3D implied volatility surface

![SPY 3D surface](demo/yfinance/yfinance_demo_surface.png)

### Smile model comparison (SVI / eSSVI / SABR)

![Smile model comparison](demo/yfinance/yfinance_demo_smiles_comparison.png)

### Model fit quality comparison

![Model fit comparison](demo/yfinance/yfinance_demo_model_comparison.png)

### Implied vol heatmap (via iv_at)

![IV heatmap](demo/yfinance/yfinance_demo_iv_heatmap.png)

### Dupire local volatility

![Dupire local vol](demo/yfinance/yfinance_demo_dupire.png)

### Portfolio Greeks (delta / gamma / vega)

![Greeks heatmap](demo/yfinance/yfinance_demo_greeks.png)

### Repair comparison (before vs after)

![Repair comparison](demo/yfinance/yfinance_demo_repair.png)

## How the modules fit

- `models/` holds the typed boundary objects: `OptionContract`, `Quote`, `ExpirySlice`, `VolSurface`.
- `pricing/` covers Black-Scholes pricing, analytic Greeks, the implied-vol solver (Newton with a Brent fallback), and Dupire local vol.
- `ingestion/` has a Yahoo Finance fetcher (with real risk-free rates), a CSV loader, and the eight-rule cleaning layer.
- `arbitrage/` holds the static-arbitrage detectors, both at the quote level and on the fitted SVI curves.
- `svi/`, `ssvi/`, `sabr/` are the three smile models and their calibrators.
- `repair/` is the orchestrator that cleans, detects, fits, and re-validates.
- `surface/` is the fitted-surface query layer: `iv_at` and portfolio Greeks.
- `dynamics.py` runs SVD-based PCA on a time series of surfaces (no sklearn).
- `backtest/` is the cross-sectional mispricing backtest.
- `viz/` holds all the matplotlib plots.

## Design conventions

A few decisions are baked into the code, worth knowing up front:

- Pydantic at the edges, dataclasses inside. Input and output types are validated Pydantic models. Everything in the hot path (Greeks, violations, fitted slices) is a frozen `@dataclass`. No Pydantic in the numerical loops.
- Report, don't raise. Detection lists every violation in one pass instead of throwing on the first. Repair tells you what it couldn't fix instead of pretending it fixed everything.
- Median forward curve. The per-expiry forward is the median of strike-level put-call parity estimates, not the mean, so one bad quote can't drag it around.
- Constrained calibration. The fit objective adds a penalty for butterfly-arbitrage violations, so the model is steered away from impossible shapes instead of minimized blindly.
- Pluggable smile models. `repair(use_ssvi=..., use_sabr=...)` switches models, and a `to_raw_svi_params()` adapter keeps the rest of the pipeline unchanged.

## What's done, and what isn't

The core engine is built and tested: ingestion, cleaning, all four arbitrage checks, three smile models, the repair loop, the query layer, Dupire, PCA, and a single-snapshot mispricing backtest. The test suite covers known-value cases, IV round-trips, synthetic violation injection, and calibration recovery.

This is not a production service yet. There's no CLI, no REST API, no database, and no dashboard. Visualization is static PNGs. The backtest is single-snapshot with a frozen-vol hedge, not a rolling daily refit (that would need a historical snapshot collector that doesn't exist yet). SABR is fit but not arbitrage-free by construction, so its residuals are reported rather than eliminated. And it's demonstrated on a handful of US equities and ETFs (SPY by default; QQQ, AAPL, MSFT, NVDA also work via `--symbol`); European options only.

## Roadmap

See `docs/roadmap.md`. Done so far: PCA dynamics, SABR, Dupire local vol, and the mispricing backtest. Next: a CLI, FastAPI + DuckDB storage, a Streamlit dashboard, and a rolling daily-refit backtest.

## Tech stack

Python 3.12+, NumPy, SciPy, Pydantic, Pytest. No sklearn (PCA uses `numpy.linalg.svd`). No web framework yet.

## References

- Gatheral, *The Volatility Surface: A Practitioner's Guide* (SVI, 2004)
- Gatheral and Jacquier, *Arbitrage-free SVI volatility surfaces* (2014)
- Hagan et al., *Managing Smile Risk* (SABR, 2002)
- Dupire, *Pricing with a Smile* (local volatility)
