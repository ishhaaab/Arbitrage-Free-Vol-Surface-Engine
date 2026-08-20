# AGENTS.md — arbfree-vol-surface

## Domain Context

Arbitrage-free implied volatility surface engine: calibrates SVI and eSSVI total-variance
parameterizations to cleaned option market quotes, detects static arbitrage violations
(parity, monotonicity, butterfly, calendar), and repairs noisy surfaces by rejecting
violating quotes and refitting.

## Structure

| Directory | Responsibility |
|---|---|
| `src/arbfree_vol/models/` | Boundary data types — `OptionContract`, `Quote`, `ExpirySlice`, `VolSurface`, BS/IV input models. All Pydantic. |
| `src/arbfree_vol/pricing/` | Black-Scholes price, Greeks (analytic), implied vol solver (Newton + Brent fallback). `local_vol.py`: Dupire local vol (`LocalVolSurface`, `dupire_at`, `dupire`). Uses float-level functions internally. |
| `src/arbfree_vol/arbitrage/` | Static arbitrage detection — quote-level checks (parity, monotonicity, butterfly, calendar, wide-spread) in `quote_detect.py`; SVI-curve-level checks (min-variance, g(k) butterfly, calendar) in `svi_detect.py`. `report.py` defines `ViolationType`/`ArbitrageViolation`/`ArbitrageReport`. |
| `src/arbfree_vol/svi/` | Raw SVI parameterization (5-param: a, b, rho, m, sigma). `model.py` has `svi_total_variance(k, ...)` and `svi_g(k, ...)`. `calibration.py` has `calibrate()` and `calibrate_constrained()`. |
| `src/arbfree_vol/ssvi/` | SSVI/eSSVI (Gatheral-Jacquier). `ssvi_w(k, theta, rho, psi)`, `essvi_w(k, theta, rho, eta, gamma)`, `gatheral_jacquier_condition(theta, rho, psi)`. `to_raw_svi_params()` maps SSVI back to raw SVI for pipeline compatibility. |
| `src/arbfree_vol/ssvi/term_structure.py` | eSSVI calendar-arb-free sequential joint fit (Hendriks & Martini 2019 Prop 3.1; Corbetta et al. 2019). `fit_ssvi_surface_sequential()`, `verify_hm_condition()`. |
| `src/arbfree_vol/ssvi/diagnostics.py` | Fallback-slice diagnostics (research tool, promoted from `scripts/diagnose_fallback_slices.py`): `run_diagnostics()` runs the full pipeline over live SPY data or the deterministic synthetic W7 fixture (vol hump + rho flip → 3 fallbacks at T=0.427/0.75/1.00), with default-seed / warm-start / random-restart fit attempts (`try_hard_constrained`, `try_warm_start`, `try_random_restarts`) and the H&M neighbour check. Numbers are optimizer- and snapshot-dependent. |
| `src/arbfree_vol/repair/` | Repair pipeline — `repair(surface, use_ssvi=False, use_sabr=False)`, `iterative_repair()`. Forward curve estimation via median put-call parity (`fwd_curve.py`). `report.py` defines `RejectedQuote`, `FittedSlice`, `FittedSSVISlice`, `FittedSABRSlice`, `RepairMetrics`, `RepairReport`. |
| `src/arbfree_vol/cli.py` + `config.py` + `time/` + `rates/` | CLI (`arbfree repair|detect|price|fetch`, `build_parser()`, `main()`), YAML config (`Config`, `load_config`), DayCount/Calendar (`ACT/365F` default, `ACT/360`, `30/360`, `USNYSE`), YieldTermStructure + FRED Treasury+SOFR curve. |
| `src/arbfree_vol/ingestion/` | Data sources — `yfinance.py:fetch_chain()` (live, with `^IRX` rates), `loader.py:load_chain_csv()`. `cleaning.py:clean_quotes()` applies 8 rejection rules with audit records. |
| `src/arbfree_vol/viz/` | Matplotlib visualization — 3D per-slice ribbons, 2D (T,k) heatmap, per-expiry smile plots, violation bar charts, repair comparison. |
| `src/arbfree_vol/variance.py` | Shared helper: `slice_total_variance()` — maps a slice to `dict[strike, w]` where `w = sigma^2 * T`. Used by both arbitrage detection and SVI fitting. |
| `src/arbfree_vol/sabr/` | SABR model (Hagan et al. 2002). `model.py`: `SABRParams`, `sabr_implied_vol()`, `sabr_total_variance()`, `to_raw_svi_params()`. `calibration.py`: `calibrate_sabr()` (per-slice, N=1 fallback + seeding). `term_structure.py`: `fit_sabr_term_structure()` -- B-spline term structure on alpha(t)/nu(t)/rho(t) + cross-slice calendar-arb SOFT penalty; empirical, not arb-free by construction. Beta is fixed-hint. |
| `src/arbfree_vol/surface/` | Fitted-surface analytics. `interpolate.py`: `FittedSurface`, `build_fitted_surface(report)`, `iv_at(fs,K,T)`, `total_variance_at(fs,K,T)`. `greeks.py`: `PortfolioGreeks`, `portfolio_greeks()`, `bucketed_greeks()`. `risk.py`: `ScenarioResult`, `spot_bump_analysis()`, `vol_bump_analysis()`, `parallel_vega_pnl()`. |
| `src/arbfree_vol/pricing/local_vol.py` | Dupire local volatility. `LocalVolSurface`, `dupire_at(fs,K,T,dT)`, `dupire(fs,strikes,maturities,dT)`. Gatheral SSVI-compatible Dupire strip-out via finite differences on the fitted surface. |
| `src/arbfree_vol/dynamics.py` | Time-series surface analysis. `SurfaceSnapshot`, `SurfaceSeries`, `PCAResult`; `fit_surface_series()`, `parameter_matrix()`, `pca_deformations()`. SVD-based PCA (no sklearn). |

Layering: `models/` is the shared kernel (no internal imports). `ingestion/` produces `VolSurface`. `arbitrage/` reads `VolSurface` and produces `ArbitrageReport`. `repair/` orchestrates ingestion → detection → forward curve → fitting → final detection. `viz/` is a pure consumer of the report types.

## Math-to-Code Mapping

- Raw SVI: `w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))` → `svi_total_variance(k, a, b, rho, m, sigma)` in `svi/model.py:38`.
- SSVI: `w(k) = (theta/2) * (1 + rho*psi*k + sqrt((psi*k+rho)^2 + (1-rho^2)))` → `ssvi_w(k, theta, rho, psi)` in `ssvi/model.py:45`.
- eSSVI: `psi = eta / theta^gamma` → `essvi_psi()` / `essvi_w()` in `ssvi/model.py:38,54`.
- No-arb density condition `g(k) >= 0` → `svi_g(k, a, b, rho, m, sigma)` in `svi/model.py:29`.
- Log-moneyness: `k = ln(K/F)` computed per-slice in the fitting paths; `F = S * exp((r-q)T)` via `_forward_price` in `svi/data.py`.
- Notation matches Gatheral (2004) for raw SVI and Gatheral & Jacquier (2014) for SSVI/eSSVI. Variable names follow the papers directly: `a, b, rho, m, sigma` for raw SVI; `theta, rho, psi, eta, gamma` for eSSVI.
- SABR Hagan asymptotic IV (Hagan et al. 2002 eq. A.69a/b) → `sabr_implied_vol(k, F, T, alpha, beta, rho, nu)` in `sabr/model.py:57`. ATM limit used for `|k| <= 1e-8`.
- Dupire local vol (Gatheral 2004 SSVI-compatible form) → `dupire_at(fs, K, T)` in `pricing/local_vol.py`. Operating on `w = sigma_imp^2 * T` from `total_variance_at`.
- Linear-T interpolation in total-variance space → `total_variance_at(fs, K, T)` / `iv_at(fs, K, T)` in `surface/interpolate.py:110,202`. Each slice uses its own forward; out-of-surface queries raise `ValueError`.

## Abstraction & Extensibility Conventions

- **Boundary vs. compute**: Pydantic `BaseModel` is used only at I/O boundaries (`OptionContract`, `Quote`, `VolSurface`, `SVIParams`, `SSVIParams`). All compute output uses frozen `@dataclass(slots=True)` — `Greeks`, `ArbitrageViolation`, `FittedSlice`, `RepairReport`. Do not introduce Pydantic into hot paths (IV solving, Greeks, SVI evaluation).
- **Smile model pluggability**: Three smile models available -- raw SVI, eSSVI (primary, arb-free by construction), and SABR (comparison, empirical). `repair(use_ssvi=False, use_sabr=False)` with mutual exclusivity between `use_ssvi` and `use_sabr`. `to_raw_svi_params()` in `ssvi/model.py:109` maps eSSVI to raw SVI; `sabr/model.py:123` maps SABR to raw SVI. Downstream pipeline (plots, arb detection) works unchanged. **eSSVI** is the arb-certified primary surface: `use_ssvi=True` fits slices sequentially by increasing maturity with the Hendriks & Martini (2019) Prop 3.1 no-calendar-spread condition enforced as a HARD optimizer constraint (non-decreasing theta and chi=theta*psi, plus |(rho*chi)_{i+1} - (rho*chi)_i| / (chi_{i+1} - chi_i) <= 1 between adjacent slices), plus both Gatheral-Jacquier (2014) butterfly bounds per slice. Per-slice rho is fully free (tanh-reparametrised per slice; NO cross-slice functional form). See `ssvi/term_structure.py:fit_ssvi_surface_sequential`. The grid-based `detect_svi_surface` is load-bearing: slices that fall back to the unconstrained fit (see `RepairReport.fallback_slices` / `repair_infeasible`) are NOT arb-free by construction, and `detect_svi_surface` reports those violations as `remaining_violations`. `FittedSSVISlice.essvi` is None on this path (the H&M construction fits per-slice phi). **SABR** is a comparison parametrisation: `use_sabr=True` fits B-spline term structures on alpha(t)/nu(t)/rho(t) across expiries with a cross-slice calendar-arb SOFT penalty (`sabr/term_structure.py:fit_sabr_term_structure`); coefficients are reparametrised at the B-spline control-point level (scaled tanh for rho, exp+floor for alpha/nu) to stay in-range between knots by the convex-hull property. Calendar-arb verification is EMPIRICAL/grid-based via `detect_svi_surface`, NOT a closed-form guarantee; dynamic SABR is not implemented. All three paths report remaining violations honestly; `RepairReport.repair_infeasible` is set True if the eSSVI hard constraints cannot be satisfied. To add a new smile model, follow the same pattern as `sabr/`: Pydantic boundary type, `model.py` formulas, `calibration.py` (and optional `term_structure.py` for cross-slice fits), `Fitted<Model>Slice` in `repair/report.py`, `to_raw_svi_params()` adapter, a `use_<model>=False` flag in `repair()`.
- **Detection entry points**: `detect(surface)` for tests/synthetic data; `detect_with_forward(surface)` for real market data (adds forward-curve pre-pass). New arb checks wire into both. `detect_svi()` for per-slice SVI checks; `detect_svi_surface()` for cross-slice (calendar).
- **Report, not raises**: Detection enumerates all violations in one pass. Repair reports honestly if violations remain.
- **Median, not mean**: Forward curve estimates use median across strike-level parity pairs so one outlier can't corrupt the estimate.
- **Per-slice r/q term structure**: Optional `risk_free`, `div_yield` on `ExpirySlice`. `get_r(surface, sl)` / `get_q(surface, sl)` falls back to surface-level. Populated automatically by `detect_with_forward()` and `repair()`.
- **Constrained calibration**: `calibrate_constrained(points, arb_penalty=100.0, prev_slice=None)` augments the least-squares residual with `sqrt(arb_penalty)*sqrt(max(-g(k), 0))` on a k-grid, `sqrt(arb_penalty)*sqrt(max(-w_min, 0))` for min-variance, and — when `prev_slice` is supplied — `sqrt(arb_penalty)*sqrt(max(w_prev(k) - w(k), 0))` for calendar arbitrage. `_fit_slice()` in `repair/engine.py` threads the previous fitted slice's params through `prev_slice` when fitting in ascending-expiry order (SVI path only; eSSVI/SABR fit per-slice without it). The unconstrained `calibrate()` is preserved for back-compat/tests.
- **Surface interpolation**: `build_fitted_surface(report)` extracts the surface from a `RepairReport`; `iv_at(K, T)` is the public entry point for IV at arbitrary strikes/maturities, used by Greek aggregation (`surface/greeks.py`), risk scenarios (`surface/risk.py`), and Dupire (`pricing/local_vol.py`).

## No-Hardcoding Rule

Model parameters, market conventions, and dataset-specific constants must be configurable — never inlined as magic numbers. Existing patterns to follow:

- **`cleaning.py:clean_quotes()`** (`src/arbfree_vol/ingestion/cleaning.py:149`): thresholds are function arguments with defaults — `min_T`, `max_spread_ratio`, `max_log_moneyness`. Each rule is a separate `_check_*()` function accepting the threshold as a parameter.
- **`implied_vol.py`** (`src/arbfree_vol/pricing/implied_vol.py:13-14`): `_NEWTON_ITERS` and `_NEWTON_TOL` as module-level constants (acceptable for fixed algorithmic params).
- **`iterative_repair(surface, max_iters=5)`**: iteration count is a function argument with a default.
- Day-count convention: `days / 365.0` used for `T` in `loader.py:32` and `yfinance.py:143` (ACT/365 fixed). This is the de facto convention; if switching to ACT/360, change the divisor.
- For new code: if a threshold appears (arb tolerance, bound, grid size), define it as a named constant at the module level or a function parameter with an explicit default. Run a grep for `1e-4` and `0.05` to find the existing tolerance pattern used across `quote_detect.py` and `svi_detect.py`.

## Numerical Rigor

- **Tolerance**: `1e-4` is the de facto arb check tolerance across `quote_detect.py` (monotonicity, butterfly, calendar) and `svi_detect.py` (min-variance, butterfly, calendar). `_NEWTON_TOL = 1e-8` for IV convergence.
- **NaN/Inf**: `implied_vol()` returns `None` for invalid prices (no root in bracket); `slice_total_variance()` drops quotes with `None` IV; `viz/` uses `np.errstate(divide="ignore", invalid="ignore")` and `np.ma.masked_invalid()`. No systematic NaN/Inf propagation guard exists across the codebase — this is a gap.
- **Zero edge cases**: `vega_floats()` checks `v <= 0.0` before Newton step; `essvi_psi()` raises `ValueError` on `theta <= 0`; near-expiry quotes are rejected in cleaning (`min_T` threshold). Zero time-to-expiry is not systematically handled in the pricing layer — gap.
- **Known-value tests**: BS price (`test_black_scholes.py`), Greeks (`test_greeks.py`), IV round-trip (`test_implied_vol.py`), SVI formulas at ATM (`test_svi.py`, `test_ssvi.py`), forward curve (`test_fwd_curve.py`). No tests validate against published benchmarks (e.g. Gatheral's example parameter sets) — gap.

## Testing Approach

- **54 test files, 681 tests** (667 passing + 14 deselected, 2026-08-20). `pytest` with `approx()` for floating-point assertions.
- **Known-value regression**: BS prices and Greeks checked against independently computed reference values (`abs=1e-4` to `1e-6`).
- **Round-trip**: IV solver tested by pricing at a known vol, then recovering the same vol from the price.
- **Synthetic surface tests**: Construction of `VolSurface` from BS-generated quotes to test arb detection on clean data; deliberate injection of violations (parity break, monotonicity break, butterfly break, calendar break) to test detection.
- **SVI/SSVI calibration recovery**: Generate points from known parameters, calibrate, assert parameter recovery and pointwise fit accuracy.
- **Property-based / sanity**: Delta parity (`delta_call - delta_put == 1`), gamma/vega equality across call/put.
- **Ingestion**: CSV loader tested with tempfiles; yfinance fetcher tested with `unittest.mock` (no network calls); cleaning rules tested per-rule and in combination.
- **Visualization**: Smoke tests only — assert `fig.axes is not None` after calling each plot function. Not validated for visual correctness.
- **Benchmark**: `bench/bench_iv.py` measures IV solver throughput (no assertions, informational only).
- **Known-value regression** (new): Local-vol Dupire flat-vol benchmark: verify `dupire` on a flat IV surface recovers the flat vol to rel=5e-3 (+ calendar-arb guard test).

## Data Flow

```
Yahoo Finance / CSV → fetch_chain() / load_chain_csv()
  → clean_quotes() → (kept, RejectedRecord[])
  → VolSurface(spot, r, q, slices)
  → detect_with_forward()
    → estimate_forward_curve() → populate_per_slice_r()
    → _check_parity(forward_price=...) / monotonicity / butterfly / calendar / wide_spread
  → repair(surface)
    → detect_with_forward() → _build_rejection_set() → _build_cleaned_surface()
    → estimate_forward_curve() → populate_per_slice_r()
    → _fit_slice() (SVI), fit_ssvi_surface_sequential() (eSSVI), or fit_sabr_term_structure() (SABR)
    → detect_svi_surface()
    → return RepairReport
```

## How to run

```
python -m pytest tests/                          # all 681 tests
python demo/yfinance/yfinance_demo.py --symbol SPY   # end-to-end SPY demo (7 plots)
python demo/essvi/essvi_demo.py                  # raw SVI vs eSSVI comparison
python demo/ticker_compare/ticker_compare.py     # cross-ticker SVI/eSSVI/SABR comparison
python bench/bench_iv.py                         # IV solver benchmark
arbfree --help                                   # CLI: repair|detect|price|fetch
```

## What NOT to do

- Do not add Pydantic to hot paths (IV solving, Greeks, SVI evaluation). Pydantic is for boundary data only.
- Do not import from `repair/` in `arbitrage/`. (`arbitrage.quote_detect` imports `repair.fwd_curve` which is OK, but `repair.report` imports `arbitrage.report` so reverse cycles will break.)
- Do not change the function signature of `detect()` (used in tests).
- Do not use the mean of strike-level forward estimates — use the median (one outlier quote can corrupt the mean).
- Do not try to make the 3D surface smooth by interpolating SVI params between slices — raw SVI params don't interpolate well physically. Show per-slice ribbons + data scatter instead.
- Do not call `calibrate_constrained` with `arb_penalty` inlined as a magic number — pass it explicitly when tuning.

## Important files

- `src/arbfree_vol/models/option.py`, `surface.py` — boundary data types
- `src/arbfree_vol/arbitrage/report.py` — `ViolationType`, `ArbitrageViolation`, `ArbitrageReport`
- `src/arbfree_vol/repair/report.py` — `RejectedQuote`, `FittedSlice`, `FittedSSVISlice`, `FittedSABRSlice`, `RepairMetrics`, `RepairReport`
- `.memories/architecture-convention.md` — package structure rationale
- `.memories/arbitrage-detection-design.md` — detection design decisions
- `.memories/svi-fitting-progress.md` — SVI/SSVI design notes
- `src/arbfree_vol/surface/interpolate.py` — `FittedSurface`, `iv_at`; the public surface-query layer.
- `src/arbfree_vol/pricing/local_vol.py` — `LocalVolSurface`, `dupire_at`; Dupire strip-out.
- `src/arbfree_vol/dynamics.py` — `SurfaceSeries`, `pca_deformations`; time-series surface analysis.

## Quick reference: common tasks

**Add a new arbitrage check:**
- Add `_check_<name>()` in `quote_detect.py`, add `ViolationType` in `report.py`, wire into `detect()` and `detect_with_forward()`, add tests in `test_arbitrage.py`.

**Add a new smile model:**
- Follow the same pattern as `sabr/`: Pydantic boundary type, `sabr/model.py` formulas, `sabr/calibration.py`, `FittedSABRSlice` in `repair/report.py`, `to_raw_svi_params()` adapter, a `use_<model>=False` flag in `repair()`. Add tests in `tests/test_<model>.py`.

**Improve the repair engine:**
- Read `src/arbfree_vol/repair/engine.py` and `.memories/svi-fitting-progress.md`. Test with `tests/test_repair_engine.py` and `python demo/yfinance/yfinance_demo.py --symbol SPY`.

**Visualize something:**
- Add a function to `src/arbfree_vol/viz/`. Use matplotlib + numpy only. Add a smoke test in `tests/test_viz.py`.
