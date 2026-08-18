# Architecture — arbfree-vol-surface

> Status: **thorough** — grounded in `src/arbfree_vol/*` as implemented. Covers components, data flow, boundaries, and planned storage/API.

## Goals

The engine turns noisy market quotes into a usable, arbitrage-aware vol surface:

1. Ingest messy chains (Yahoo Finance / CSV).
2. Clean with auditable, deterministic rules.
3. Invert Black-Scholes to IV (`pricing/implied_vol.py`).
4. Detect static arbitrage honestly — enumerate every violation.
5. Fit a smile model (raw SVI / eSSVI / SABR) and repair the surface.
6. Expose the result for analytics, risk, and viz — with honest failure reporting.

Correctness, testability, and `RepairReport.repair_infeasible` honesty beat demo surface area.

## System Components

| Directory | Responsibility | Key entry points |
|---|---|---|
| `src/arbfree_vol/models/` | Shared kernel — boundary types. No internal imports. | `models/surface.py:5` `Quote`, `models/surface.py:18` `ExpirySlice`, `models/surface.py:24` `VolSurface`, `models/option.py:6` `OptionType`, `models/option.py:11` `OffendingQuote`, `models/option.py:23` `OptionContract` |
| `src/arbfree_vol/ingestion/` | Data sources → `VolSurface` | `ingestion/loader.py:load_chain_csv()`, `ingestion/yahoo.py:fetch_chain()` (with `^IRX` rates), `ingestion/cleaning.py:149` `clean_quotes()` |
| `src/arbfree_vol/data/` | Pre-ingestion DataFrame filter (before `VolSurface` exists) | `data/quality.py:filter_option_chain()` → `DropRecord[]` |
| `src/arbfree_vol/pricing/` | Black-Scholes, Greeks, IV solver, Dupire | `pricing/black_scholes.py`, `pricing/implied_vol.py:13` (`_NEWTON_TOL=1e-8`, `None` on invalid price), `pricing/greeks.py`, `pricing/local_vol.py:dupire_at` |
| `src/arbfree_vol/variance.py` | Shared helper `w = σ²T` | `variance.py:slice_total_variance()` — drops `None` IVs, averages call/put at same strike |
| `src/arbfree_vol/forward.py` | Forward curve (shared by `arbitrage` + `repair`) | `forward.py:estimate_forward_curve()` (median parity), `forward.py:populate_per_slice_r()` |
| `src/arbfree_vol/arbitrage/` | Static arb detection → `ArbitrageReport` | `arbitrage/report.py:6` `ViolationType`, `arbitrage/report.py:15` `ArbitrageViolation`, `arbitrage/quote_detect.py:detect()`, `arbitrage/quote_detect.py:detect_with_forward()`, `arbitrage/svi_detect.py:detect_svi()`, `arbitrage/svi_detect.py:detect_svi_surface()` |
| `src/arbfree_vol/svi/` | Raw SVI 5-param `w(k)` + calibration | `svi/model.py:38` `svi_total_variance()`, `svi/model.py:29` `svi_g()`, `svi/calibration.py:calibrate()`, `svi/calibration.py:calibrate_constrained()` |
| `src/arbfree_vol/ssvi/` | SSVI/eSSVI (Gatheral-Jacquier) | `ssvi/model.py:45` `ssvi_w()`, `ssvi/model.py:54` `essvi_w()`, `ssvi/model.py:109` `to_raw_svi_params()`, `ssvi/term_structure.py:fit_ssvi_surface_sequential()` (H&M Prop 3.1 hard constraints) |
| `src/arbfree_vol/sabr/` | SABR Hagan + B-spline term structure | `sabr/model.py:57` `sabr_implied_vol()`, `sabr/model.py:123` `to_raw_svi_params()`, `sabr/term_structure.py:fit_sabr_term_structure()` |
| `src/arbfree_vol/repair/` | Orchestrator `VolSurface → RepairReport` | `repair/engine.py:repair(use_ssvi, use_sabr)`, `repair/report.py:33` `RepairReport`, `repair/report.py:9` `RejectedQuote` |
| `src/arbfree_vol/surface/` | Fitted-surface interpolation + risk | `surface/interpolate.py:build_fitted_surface()`, `surface/interpolate.py:iv_at()`, `surface/interpolate.py:total_variance_at()`, `surface/greeks.py`, `surface/risk.py` |
| `src/arbfree_vol/dynamics.py` | Time-series surface PCA | `dynamics.py:fit_surface_series()`, `dynamics.py:pca_deformations()` (SVD, no sklearn) |
| `src/arbfree_vol/viz/` | Pure consumers of reports | `viz/surface.py`, `viz/smiles.py`, `viz/violations.py`, `viz/comparison.py` (matplotlib only) |
| `src/arbfree_vol/storage/` | **Planned M6** — DuckDB persistence | `storage/duckdb_store.py` (stub) |
| `src/arbfree_vol/api/` | **Planned M6** — FastAPI | `api/main.py`, `api/routes.py` (stub) |

`models/surface.py:31,41` `get_r()` / `get_q()` resolve per-slice term structure (prefer `sl.risk_free`/`sl.div_yield`, fall back to `surface.risk_free`/`div_yield`).

## End-to-End Data Flow

```
Yahoo Finance / OpenBB / CSV
  │  ingestion/yahoo.py:fetch_chain()  ·  ingestion/loader.py:load_chain_csv()
  │  data/quality.py:filter_option_chain()  (OI ≥ 10, spread ≤ 50% → DropRecord[])
  ▼
VolSurface  { spot, risk_free, div_yield, slices: ExpirySlice[] }   ← models/surface.py:24
  │  ingestion/cleaning.py:clean_quotes()  — 8 rules, first-match per quote
  │    → (kept Quote[], RejectedRecord[])    negative/zero price, crossed, wide spread,
  │                                          stale, illiquid, deep OTM/ITM, min_T
  ▼
Arbitrage Detection  — arbitrage/quote_detect.py
  │  forward.py:estimate_forward_curve() → dict[T, F]  (median parity)
  │  forward.py:populate_per_slice_r()  (threads F-implied r into each ExpirySlice)
  │  _check_parity / _check_monotonicity / _check_butterfly / _check_calendar / _check_wide_spread
  │  arbitrage/svi_detect.py:detect_svi_surface() for fitted-curve checks
  │    → ArbitrageReport { violations: ArbitrageViolation[], is_arbitrage_free }
  ▼
Repair  — repair/engine.py:repair(surface, use_ssvi, use_sabr)  [7 steps]
  │  1  _detect_violations()          → ArbitrageReport
  │  2  _build_rejection_set()         → (reject_set, RejectedQuote[])
  │  3  _build_cleaned_surface()       → VolSurface | None
  │  4  estimate_forward_curve() + populate_per_slice_r() on cleaned surface
  │  5  strategy.fit()  (dispatched via repair/strategies/)
  │  │     SVI:   svi/calibration.py:calibrate_constrained(prev_slice=…)  (soft arb penalty)
  │  │     eSSVI: ssvi/term_structure.py:fit_ssvi_surface_sequential()    (H&M hard constraints)
  │  │     SABR:  sabr/term_structure.py:fit_sabr_term_structure()        (B-spline + soft penalty)
  │  6  arbitrage/svi_detect.py:detect_svi_surface()  → remaining_violations
  │  7  RepairMetrics + _consolidate_failures()
  │    → RepairReport
  ▼
Fitted Surface  — RepairReport  (repair/report.py:33)
  ├─ cleaned_surface: VolSurface | None
  ├─ fitted_slices: tuple[FittedSlice, ...]           (raw SVI — common currency)
  ├─ fitted_ssvi_slices / fitted_sabr_slices         (native params, if selected)
  ├─ remaining_violations: ArbitrageReport            (grid-based post-fit check)
  ├─ repair_infeasible: bool  ·  fallback_slices: list[float]  ·  failed_slices
  └─ metrics: RepairMetrics { n_rejected, rejection_rate, … }
  │
  ├─► viz/  — 3D ribbons, 2D (T,k) heatmap, smile-by-expiry, violation bars,
  │           raw-vs-repaired comparison, Dupire heatmap
  ├─► surface/interpolate.py:FittedSurface + iv_at(K,T) / total_variance_at(K,T)
  │           → surface/greeks.py, surface/risk.py, pricing/local_vol.py
  └─► storage/ (planned) + api/ (planned)  → DuckDB + FastAPI
```

Property: **report, not raises.** Detection enumerates all violations in one pass; `RepairReport` always carries `remaining_violations` and `repair_infeasible` so callers handle partial failure without exceptions.

## Module Boundaries & Dependency Rules

DAG rooted at `models/` — no cycles:

```
models/  (no internal imports — safe to import anywhere)
  ↑
  ├── ingestion/        produces  VolSurface
  ├── pricing/          reads     VolSurface / OptionContract → price / IV / Greeks
  ├── variance.py / forward.py   reads VolSurface → w / forward curve
  ├── arbitrage/        reads     VolSurface (+ forward) → ArbitrageReport
  │     └── imports forward.py only — never repair/  (forward.py lives at top level to break the cycle)
  ├── svi/ ssvi/ sabr/  reads     (k,w) points → params  (each has to_raw_svi_params() adapter)
  ├── repair/           orchestrates VolSurface → RepairReport
  │     └── imports arbitrage/report.py, models/*, forward.py, strategies/*
  │         strategies/* own all per-model fitting imports; engine.py imports no model directly
  │         get_strategy(use_ssvi, use_sabr) enforces mutual exclusivity
  ├── surface/          reads     RepairReport → FittedSurface → iv_at(K,T)
  └── viz/              consumes  ArbitrageReport / RepairReport / FittedSurface  (no business logic)
```

Invariants (enforced by convention — `docs/AGENTS.md`):

* `arbitrage/` never imports `repair/` (except shared `forward.py`/`variance.py` which are top-level for that reason).
* `repair/report.py` imports `arbitrage/report.py`, not the reverse.
* Adding a new smile model = copy the `sabr/` pattern: Pydantic boundary type → `model.py` formulas → `calibration.py` (+ optional `term_structure.py`) → `Fitted<Model>Slice` in `models/fitted.py` / `repair/report.py` → `to_raw_svi_params()` → `use_<model>` flag in `repair()`.

## Design Decisions

**Pydantic at boundaries, frozen dataclasses on hot paths.**
`OptionContract`, `Quote`, `VolSurface`, `ExpirySlice`, `SVIParams`, `SSVIParams` are Pydantic `BaseModel` — validation at I/O (CSV/yfinance) with clear errors. Compute outputs (`Greeks`, `ArbitrageViolation`, `FittedSlice`, `RepairReport`, `OffendingQuote` in `models/option.py:11`) are `@dataclass(frozen=True, slots=True)` — cheap, immutable, hashable. Never add Pydantic inside IV solving, Greeks, or SVI evaluation.

**Smile model pluggability.**
Three models behind one orchestrator: raw SVI (default), eSSVI (`use_ssvi=True`, primary arb-free path), SABR (`use_sabr=True`, empirical comparison). Each exposes `to_raw_svi_params()` (`ssvi/model.py:109`, `sabr/model.py:123`) so downstream detection, interpolation, and viz work on raw SVI uniformly. `repair()` enforces `use_ssvi`/`use_sabr` mutual exclusivity and delegates to `repair/strategies/`.

* eSSVI: sequential by `T` with Hendriks & Martini (2019) Prop 3.1 as **hard** optimizer constraints (non-decreasing `theta`, `chi=theta·psi`, and `|(rho·chi)ᵢ₊₁-(rho·chi)ᵢ|/(chiᵢ₊₁-chiᵢ) ≤ 1`) plus both Gatheral-Jacquier (2014) butterfly bounds per slice. `rho` is per-slice free (tanh-reparametrised). Slices that cannot satisfy the hard constraints fall back to unconstrained per-slice fit and are surfaced as `RepairReport.fallback_slices` / `repair_infeasible=True` — then `detect_svi_surface` reports them as `remaining_violations`.
* SABR: cubic B-spline term structures on `alpha(t)/nu(t)/rho(t)` with cross-slice calendar **soft** penalty; coefficients reparametrised at control points (`tanh` for `rho`, `exp+floor` for `alpha`/`nu`) to stay in-range by the B-spline convex-hull property. Calendar verification is empirical/grid-based, not a closed-form guarantee.

**Report, not raises.**
`detect()` / `detect_with_forward()` return `ArbitrageReport` with every violation; `repair()` always returns `RepairReport` with `remaining_violations` and `repair_infeasible`. Callers branch on data instead of catching exceptions. Slices that fail both fits appear in `failed_slices`.

**Median, not mean for forwards.**
`forward.py:estimate_forward_curve()` computes `F = e^{rT}(C-P)+K` per strike-level put-call pair and takes the **median** across strikes. One outlier quote cannot corrupt `F`; `populate_per_slice_r()` then back-fills `sl.risk_free = ln(F/S)/T + q`.

**Per-slice `r/q` term structure.**
`VolSurface` carries surface-level `risk_free`/`div_yield`, but `ExpirySlice` has optional `risk_free`/`div_yield` overrides (`models/surface.py:21,22`). `get_r`/`get_q` (`models/surface.py:31,41`) prefer the per-slice value. `detect_with_forward()` and `repair()` populate these via `populate_per_slice_r`. Callers must use `get_r`/`get_q`, not `surface.risk_free` directly.

**Constrained calibration is explicit.**
`svi/calibration.py:calibrate_constrained(points, arb_penalty=100.0, prev_slice=None)` augments the least-squares residual with `sqrt(arb_penalty)·sqrt(max(-g(k),0))` (butterfly), `sqrt(arb_penalty)·sqrt(max(-w_min,0))` (min-variance), and when `prev_slice` is given `sqrt(arb_penalty)·sqrt(max(w_prev(k)-w(k),0))` (calendar). `repair/engine.py:_fit_slice()` threads `prev_slice` in ascending-`T` order for the raw-SVI path only; eSSVI/SABR use their own term-structure fitters. `calibrate()` (unconstrained) is preserved for tests/back-compat. Never inline `arb_penalty` as a magic number.

## Storage Design (Planned — M6)

Not yet implemented. Intended design (`docs/Project.md` Phase 5, `docs/roadmap.md` M6):

* **DuckDB** — embedded, columnar, no server; ideal for single-user research and interview demos. Preferred over PostgreSQL for the first iteration.
* Tables keyed by `chain_id`/`surface_id`: `raw_chains`, `cleaned_chains`, `fitted_surfaces`, `repair_reports`. Fitted slices and rejected quotes stored as JSON columns from their Pydantic/dataclass representations.
* The storage layer will depend on `models/` and `repair/report.py`, not the reverse. `VolSurface` and `RepairReport` serialize via their existing types.

## API Design (Planned — M6)

Not yet implemented. Thin HTTP wrapper around the existing `repair()`/`detect()` functions — no business logic duplicated in handlers. Request/response schemas are Pydantic, mapped to `models/*` types (`docs/Project.md`):

```
GET  /health
POST /chains/ingest                → { chain_id }
GET  /chains/{chain_id}
POST /chains/{chain_id}/clean      → { kept, rejected }
POST /chains/{chain_id}/implied-vols
POST /chains/{chain_id}/arbitrage-check  → ArbitrageReport
POST /chains/{chain_id}/fit-svi    → { fitted_slices }
POST /chains/{chain_id}/repair     → RepairReport
GET  /surfaces/{surface_id}
GET  /surfaces/{surface_id}/violations
```

Added only after the quant engine is well-tested (current: 569 tests via `pytest`).

## Why Not Notebook-Only

| Concern | Notebook-only failure | Package answer |
|---|---|---|
| **Boundaries** | Ingestion + cleaning + pricing + fitting share one namespace | `models → ingestion → arbitrage → repair → viz` is an import-checkable DAG |
| **Testability** | Cells are not importable | `tests/test_black_scholes.py`, `test_implied_vol.py`, `test_arbitrage.py`, `test_svi.py`, `test_repair_engine.py` — known-value, round-trip, synthetic-injection, calibration-recovery tests |
| **Reusability** | `forward.py` / `variance.py` get copy-pasted across cells | Shared helpers imported by both detection and repair |
| **Honest failure** | Plotting `0.0` for negative `w` hides failure | `RepairReport.repair_infeasible` / `fallback_slices` / `remaining_violations` make infeasibility visible |
| **Interview signal** | One-file exploration | `pyproject.toml` + `ruff` + `pyright` (standard) + `pytest --cov-fail-under=85` + typed boundaries + frozen compute types reads as production engineering |

Notebooks remain for exploration (`notebooks/surface_research.ipynb`, planned `notebooks/full_pipeline.ipynb` — M2), but the engine lives in `src/arbfree_vol/`.
