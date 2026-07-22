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

**Status:** Not started (deprioritized; in favor of CLI/SABR/Dupire/PCA work in milestones 3/5/7)

Replace static Matplotlib figures with an interactive interface.

### Features

- Upload CSV or fetch data from `yfinance`
- Interactive 3D volatility surface
- 2D heatmap
- Hover inspection showing:
  - log-moneyness (`k`)
  - maturity (`T`)
  - implied volatility
- Toggle between SVI and eSSVI
- Interactive cleaning threshold sliders

### Files

- `dashboard/app.py`
- `dashboard/requirements.txt`

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

**Status:** Not started

### Goals

- Cross-sectional mispricing detection: at each daily snapshot, mark every market quote against the fitted arb-free surface and flag quotes where `|market_IV - model_IV| > threshold`.
- Threshold-based trading strategy: long underpriced, short overpriced options (delta-hedged) and hold-to-expiry.
- Backtest metrics: Sharpe ratio, hit rate, max drawdown, distribution of P&L per trade.
- Realized P&L against actual SPY price path for held-to-expiry trades.
- Data dependency note: requires historical SPY option snapshots + the underlying price path. Can be approximated with a single-snapshot rolling study when historical chains are unavailable.

### Files

- `arbfree_vol/backtest/`
- `notebooks/backtest_demo.ipynb`

---

# Progress Summary

| Milestone | Status |
|-----------|--------|
| CLI + Configuration | Not started |
| Notebook | Not started |
| Surface Dynamics (PCA) | Completed |
| Interactive Dashboard | Not started (deprioritized) |
| SABR | Completed |
| FastAPI + DuckDB | Not started |
| Local Volatility (Dupire) | Completed |
| Documentation | Not started |
| Mispricing Backtest | Not started |