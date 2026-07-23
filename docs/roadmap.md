# Project Roadmap

This document tracks planned improvements for the project. Each milestone represents a major feature or development phase.

---

## Milestone 1 — CLI Entry Point + Configuration

**Status:** Not started

Make the system feel like a real quantitative finance tool.

### Goals

- `arbfree repair chain.csv --spot 450 --use-ssvi --plot`
- `arbfree detect chain.csv`
- `arbfree price --strike 100 --spot 100 --vol 0.2`
- YAML configuration file (`config.yaml`) containing:
  - default thresholds
  - interest rates
  - cleaning parameters
- CLI reads the configuration automatically, with command-line flags overriding defaults.

### Files

- `examples/arbfree`
- `arbfree_vol/cli.py`
- `config.yaml`
- `arbfree_vol/config.py`

---

## Milestone 2 — End-to-End Jupyter Notebook

**Status:** Not started

A complete walkthrough of the entire pipeline using SPY options data.

### Sections

1. Fetch SPY option chain (`yfinance`)
2. Clean data and inspect rejected quotes
3. Detect arbitrage violations on raw data
4. Repair pipeline
   - reject quotes
   - build forward curve
   - fit SVI
   - validate arbitrage constraints
5. Compare raw vs repaired violation counts
6. Visualize
   - volatility surface
   - smiles
   - heatmaps
7. Experiment with eSSVI on long-dated expiries

### File

- `notebooks/full_pipeline.ipynb`

---

## Milestone 3 — Surface Dynamics (PCA / Time Series)

**Status:** Completed

Study how the volatility surface evolves through time.

### Goals

- Fetch ~30 days of SPY options
- Fit an SVI surface for each day
- Track parameter evolution
- Perform PCA on parameter changes
- Visualize dominant deformation modes:
  - Level
  - Tilt
  - Curvature
  - Shift
  - Wing movement
- Estimate volatility-of-volatility from PCA scores

### Files

- `arbfree_vol/dynamics.py`
- `notebooks/surface_dynamics.ipynb`

### Implementation notes

- `dynamics.py`: `SurfaceSnapshot`, `SurfaceSeries`, `PCAResult` dataclasses; `fit_surface_series()`, `parameter_matrix()`, `pca_deformations()` — SVD-based PCA (no sklearn dependency). New tests: `tests/test_dynamics.py`.

---

## Milestone 4 — Interactive Dashboard

**Status:** Not started

Refreshed plan: Streamlit-based single-page app.

### Features

- Fetch SPY options from yfinance with a click, or upload CSV
- Toggle between SVI / eSSVI / SABR with radio buttons
- Interactive 3D Plotly surface with hover tooltips (k, T, IV, Dupire)
- 2D heatmap with hover inspection
- Sliders for spot bump scenarios with live P&L bar chart
- Greeks heatmap with dropdown to select Greek
- Auto-refresh button to pull fresh market data

### Files

- `dashboard/app.py`
- `dashboard/requirements.txt` (streamlit, plotly)

---

## Milestone 5 — Additional Smile Models (SABR)

**Status:** Completed

Implement the SABR volatility model, commonly used in FX and fixed income.

### Goals

- SABR parameterization
  - α (alpha)
  - β (beta)
  - ρ (rho)
  - ν (vol-of-vol)
- Hagan et al. approximation
- Calibration using `scipy.optimize.least_squares`

### Files

- `arbfree_vol/sabr/`
- `tests/test_sabr.py`

### Implementation notes

- SABR Hagan (2002) asymptotic implied vol (`sabr_implied_vol`), `to_raw_svi_params` adapter, `calibrate_sabr()`, `repair(use_sabr=True)` flag. New tests: `tests/test_sabr.py`. Extended: `tests/test_repair_engine.py`.

---

## Milestone 6 — FastAPI + DuckDB

**Status:** Not started

Introduce persistence and REST APIs.

### Goals

Store:

- raw option chains
- cleaned datasets
- fitted volatility surfaces
- repair reports

Expose endpoints:

- `GET /health`
- `POST /chains/ingest`
- `POST /chains/{id}/repair`
- `GET /surfaces/{id}`

Use Pydantic request/response models.

### Files

- `arbfree_vol/api/`
- `arbfree_vol/storage/`

---

## Milestone 7 — Local Volatility (Dupire)

**Status:** Completed

Extract a local volatility surface from the fitted implied volatility surface.

### Goals

- Implement Dupire's formula
- Produce local volatility surfaces
- Add comprehensive tests

### Files

- `arbfree_vol/pricing/local_vol.py`
- `tests/test_local_vol.py`

### Implementation notes

- `pricing/local_vol.py`: `LocalVolSurface`, `dupire_at()` using Gatheral SSVI-compatible Dupire strip-out via finite differences on the fitted surface; calendar-arb guard. New tests: `tests/test_local_vol.py`.

---

## Milestone 8 — Documentation

**Status:** Not started

Write the core project documentation.

### Documents

- `docs/architecture.md`
- `docs/no_arbitrage_conditions.md`
- `docs/svi.md`
- `docs/data_cleaning.md`

---

## Milestone 9 — Mispricing Backtest (Cross-Sectional Signal Research)

**Status:** Completed (Design B — single-snapshot cohort)

### Goals

- Cross-sectional mispricing detection: at each daily snapshot, mark every market quote against the fitted arb-free surface and flag quotes where `|market_IV - model_IV| > threshold`.
- Threshold-based trading strategy: long underpriced, short overpriced options (delta-hedged) and hold-to-expiry.
- Backtest metrics: Sharpe ratio, hit rate, max drawdown, distribution of P&L per trade.
- Realized P&L against actual SPY price path for held-to-expiry trades.
- Data dependency note: requires historical SPY option snapshots + the underlying price path. Can be approximated with a single-snapshot rolling study when historical chains are unavailable.

### Files

- `arbfree_vol/backtest/`
- `src/arbfree_vol/viz/backtest.py`
- `examples/backtest_demo.py`

### Implementation notes

- Single-cohort frozen-vol delta-hedge backtest (`arbfree_vol/backtest/`). Package modules: `types.py` (`MispricingSignal`, `Trade`, `TradePnL`, `BacktestResult`), `signal.py` (`detect_mispricing`), `pnl.py` (`realize_trade_pnl` — frozen-vol daily delta hedge), `prices.py` (`fetch_underlying_path`), `engine.py` (`run_backtest` — orchestration + aggregation).
- Signal detection: `|market_IV - model_IV| > threshold` (default 1 vol point). Side = +1 (long underpriced), -1 (short overpriced).
- Metrics: Sharpe (per-trade mean / std), hit rate, max drawdown (peak-to-trough on expiry-ordered cumulative P&L), P5/P50/P95 percentiles.
- `BacktestResult` is a frozen `@dataclass(slots=True)` with `trades` and `pnls` as tuples.
- Visualization: `viz/backtest.py` — 4 figures: P&L distribution histogram, cumulative P&L with max-drawdown highlight, mispricing-vs-P&L scatter by side, summary metrics bar chart (2×2 grid).
- Live SPY demo: `examples/backtest_demo.py` (yfinance → repair → run_backtest → 4 PNGs).
- All viz functions handle `n_trades == 0` gracefully (render "No trades" text).
- 5 new smoke tests in `tests/test_viz.py` (4 normal + 1 empty-result).

---

## Milestone 10 — Interactive Dashboard (Streamlit)

**Status:** Not started

Replace the static matplotlib PNG output with an interactive web app.

### Features

- Fetch SPY options from yfinance with a click, or upload CSV
- Toggle between SVI / eSSVI / SABR with radio buttons
- Interactive 3D Plotly surface with hover tooltips (k, T, IV, Dupire)
- 2D heatmap with hover inspection
- Sliders for spot bump scenarios with live P&L bar chart
- Greeks heatmap with dropdown to select Greek
- Auto-refresh button to pull fresh market data

### Files

- `dashboard/app.py`
- `dashboard/requirements.txt` (streamlit, plotly)

---

## Milestone 11 — Snapshot Collector for Real-Data PCA

**Status:** Not started

Build a lightweight data collector to accumulate real option-chain
snapshots over time for surface-dynamics PCA.

### Goals

- `collect_snapshot(symbol)` — fetch current chain, run repair, save the fitted-slice params + date to a local DuckDB or CSV
- Run once daily (cron / Task Scheduler) for 30+ days
- After sufficient snapshots, run `dynamics.pca_deformations` on real data
- Produce a real-data PCA decomposition plot alongside the synthetic demo
- Compare explained-variance ratios against the synthetic benchmark

### Files

- `arbfree_vol/collect/snapshot.py`
- `notebooks/real_pca_demo.ipynb`

---

## Milestone 12 — Rolling Backtest with Daily Refit

**Status:** Not started

Extend the single-cohort backtest to a true rolling daily-refit design.

### Goals

- True rolling entry backtest with daily surface refit.
- Requires historical option-chain snapshots (blocked on M11 snapshot collector or a paid data source).
- Compare frozen-vol hedge vs daily-refit hedge.
- Multi-cohort Sharpe / drawdown / P&L distribution aggregated across all entry dates.
- Files: `arbfree_vol/backtest/rolling.py`, `notebooks/rolling_backtest_demo.ipynb`.

### Files

- `arbfree_vol/backtest/rolling.py`
- `notebooks/rolling_backtest_demo.ipynb`

---

# Progress Summary

| Milestone | Status |
|-----------|--------|
| CLI + Configuration | Not started |
| Notebook | Not started |
| Surface Dynamics (PCA) | Completed |
| Interactive Dashboard | Not started |
| SABR | Completed |
| FastAPI + DuckDB | Not started |
| Local Volatility (Dupire) | Completed |
| Documentation | Not started |
| Mispricing Backtest | Completed (Design B) |
| Interactive Dashboard (Streamlit) | Not started |
| Snapshot Collector (Real-Data PCA) | Not started |
| Rolling Backtest (Daily Refit) | Not started |