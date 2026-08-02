# arbfree-vol-surface

An arbitrage-free implied volatility surface engine for serious quant finance research and production-style system design.

## Why This Project Exists

Most beginner finance projects stop at downloading option chains, plotting implied volatility, or pricing a vanilla option with Black-Scholes. This project is designed to go much deeper.

Raw option markets are noisy. Bid/ask spreads, stale quotes, illiquid strikes, crossed markets, bad prints, and sparse expiries can produce implied volatility surfaces that violate basic no-arbitrage conditions. A professional-grade volatility surface cannot simply interpolate those quotes. It has to clean the market data, detect violations, fit a stable model, and repair the surface into something usable for pricing and risk.

The goal of `arbfree-vol-surface` is to build that system.

This project should signal:

- Strong mathematical finance knowledge.
- Numerical methods and optimization skill.
- Understanding of options, implied volatility, and no-arbitrage.
- Clean backend architecture.
- Production-quality Python engineering.
- Ability to turn messy financial data into a robust model.

## Core Thesis

Build a system that:

1. Ingests option chain data.
2. Cleans and validates market quotes.
3. Computes implied volatilities from option prices.
4. Detects static arbitrage violations.
5. Fits SVI-style volatility surfaces.
6. Repairs noisy surfaces under no-arbitrage constraints.
7. Exposes the result through a clean API and later a dashboard.

This is not a stock prediction project. It is a volatility modeling and financial infrastructure project.

## Preferred Language And Stack

The project should be built in Python first.

Python is the right primary language because the hard parts are numerical finance, optimization, data cleaning, model fitting, and research iteration. The Python ecosystem is ideal for this.

Recommended stack:

- Python for the quant engine.
- NumPy and SciPy for numerical methods and optimization.
- Pandas or Polars for tabular data processing.
- DuckDB or PostgreSQL for persistence.
- Pydantic for typed data models.
- FastAPI for the API layer.
- Pytest for tests.
- Ruff for linting and formatting.
- Mypy or Pyright for type checking.
- Plotly or a TypeScript frontend later for visualizations.

Recommended project direction:

1. Python package and tests.
2. FastAPI service and DuckDB storage.
3. Visualization dashboard.
4. Optional Rust or C++ acceleration for selected numerical routines.

Do not start with C++ or Rust. Those can be added later for performance signal, but the first priority is correctness, clarity, and mathematical depth.

## System Architecture

Suggested module structure:

```text
arbfree-vol-surface/
  pyproject.toml
  README.md
  docs/
    architecture.md
    no_arbitrage_conditions.md
    svi.md
    data_cleaning.md
  src/
    arbfree_vol/
      __init__.py
      models/
        __init__.py
        option.py
        surface.py
        violations.py
      ingestion/
        __init__.py
        loader.py
        cleaning.py
      pricing/
        __init__.py
        black_scholes.py
        implied_vol.py
        greeks.py
      arbitrage/
        __init__.py
        strike_checks.py
        calendar_checks.py
        parity_checks.py
        density_checks.py
      fitting/
        __init__.py
        svi.py
        calibration.py
      repair/
        __init__.py
        constraints.py
        repair_engine.py
      storage/
        __init__.py
        duckdb_store.py
      api/
        __init__.py
        main.py
        routes.py
  tests/
    test_black_scholes.py
    test_implied_vol.py
    test_arbitrage_checks.py
    test_svi.py
    test_repair_engine.py
  notebooks/
    surface_research.ipynb
  demo/
    yfinance/yfinance_demo.py
    essvi/essvi_demo.py
    ticker_compare/ticker_compare.py
    backtest/backtest_demo.py
  examples/
    sample_chain.csv
```

The code should be structured as a real package, not as a notebook-only project.

## Core Features

### 1. Option Chain Ingestion

The system should ingest option chain data for one or more liquid underlyings.

Each option quote should include:

- Underlying symbol.
- Observation timestamp.
- Expiry date.
- Strike.
- Option type: call or put.
- Bid.
- Ask.
- Mid.
- Last price, if available.
- Volume.
- Open interest.
- Underlying spot price.
- Risk-free rate estimate.
- Dividend yield or forward estimate, if available.

The ingestion layer should normalize raw input into typed internal models.

### 2. Quote Cleaning

Raw option chains are messy. The cleaning layer should remove or flag:

- Negative prices.
- Zero or missing bids and asks.
- Crossed markets where bid is greater than ask.
- Quotes with extremely wide spreads.
- Stale contracts.
- Illiquid options with no volume and low open interest.
- Deep ITM or OTM options where IV inversion is unstable.
- Options with too little time to expiry.
- Prices violating simple intrinsic value bounds.

The system should preserve rejected quotes with rejection reasons when possible. This is important for auditability.

### 3. Black-Scholes Pricing

Implement Black-Scholes pricing for European calls and puts.

Inputs:

- Spot price `S`.
- Strike `K`.
- Time to expiry `T`.
- Risk-free rate `r`.
- Dividend yield `q`.
- Volatility `sigma`.
- Option type.

Outputs:

- Option price.
- Delta.
- Gamma.
- Vega.
- Theta.
- Rho.

This layer should have strong unit tests.

### 4. Implied Volatility Solver

Given a market option price, solve for the implied volatility that reproduces that price under Black-Scholes.

Requirements:

- Robust handling of invalid prices.
- Bracketing method such as Brent's method.
- Edge-case handling for near-zero time to expiry.
- Bounds checking against intrinsic value and maximum theoretical price.
- Return structured failure reasons instead of crashing.

The IV solver should be tested against known prices and round-trip cases:

1. Choose a volatility.
2. Price the option.
3. Solve implied volatility from the price.
4. Confirm the recovered volatility is close to the original.

### 5. Static Arbitrage Detection

The arbitrage detector is one of the most important parts of the project.

It should detect violations such as:

#### Strike Monotonicity

For a fixed expiry:

- Call prices should generally decrease as strike increases.
- Put prices should generally increase as strike increases.

Violations suggest bad quotes or arbitrage opportunities.

#### Butterfly Arbitrage

For a fixed expiry, option prices across strikes should be convex.

For calls, the second finite difference across strikes should be non-negative after accounting for uneven strike spacing. A convexity violation implies a negative risk-neutral density in that region.

#### Calendar Arbitrage

Total variance should generally be non-decreasing with maturity for a fixed log-moneyness region:

```text
w(k, T) = sigma_imp(k, T)^2 * T
```

If longer maturity total variance is lower than shorter maturity total variance at the same moneyness, the surface may contain calendar arbitrage.

#### Put-Call Parity

For European options:

```text
C - P = S * exp(-qT) - K * exp(-rT)
```

Large parity breaks may indicate bad quotes, borrow issues, dividends, or American exercise effects.

#### Negative Implied Density

The second derivative of call price with respect to strike is related to the risk-neutral density:

```text
d2C/dK2 = exp(-rT) * f_Q(K)
```

Negative density estimates imply arbitrage or numerical instability.

### 6. SVI Surface Fitting

Fit SVI total variance slices by expiry.

The raw SVI parameterization:

```text
w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
```

where:

- `w(k)` is total implied variance.
- `k = log(K / F)` is log-moneyness.
- `F` is the forward price.
- `a` controls the vertical level.
- `b` controls the slope.
- `rho` controls skew orientation.
- `m` shifts the smile horizontally.
- `sigma` controls curvature around the minimum.

Reasonable constraints:

- `b >= 0`
- `-1 < rho < 1`
- `sigma > 0`
- `w(k) > 0`

The fitting layer should calibrate SVI parameters to observed total variances while penalizing unstable or unrealistic shapes.

### 7. Arbitrage Repair Engine

The repair engine should produce a cleaned volatility surface that is close to observed market quotes while reducing or eliminating arbitrage violations.

Possible repair strategies:

1. Remove clearly bad quotes and refit.
2. Fit SVI with parameter constraints.
3. Penalize convexity and calendar violations in the objective.
4. Enforce monotonic total variance across maturities.
5. Project noisy prices onto a no-arbitrage feasible region.

The repaired surface should produce:

- Clean implied volatility grid.
- Fitted SVI parameters.
- List of excluded quotes.
- List of remaining violations, if any.
- Quality metrics comparing raw and repaired surfaces.

The repair system should be honest. If a surface cannot be fully repaired under current assumptions, it should say so.

The implementation (`repair()` in `src/arbfree_vol/repair/engine.py`) fits three smile models -- raw SVI, eSSVI (primary), and SABR (comparison) -- selectable via `use_ssvi` / `use_sabr` (mutually exclusive). The raw-SVI path applies a calendar-arbitrage SOFT penalty that threads the previous slice's fitted parameters when fitting in ascending-expiry order (commit 8b3e149). The **eSSVI** path is the arbitrage-certified primary surface: it fits SSVI slices sequentially by increasing maturity with the Hendriks & Martini (2019) Prop 3.1 no-calendar-spread condition enforced as a HARD optimizer constraint -- non-decreasing theta and chi=theta*psi across slices, plus |(rho*chi)_{i+1} - (rho*chi)_i| / (chi_{i+1} - chi_i) <= 1 between adjacent slices -- together with both Gatheral-Jacquier (2014) butterfly bounds per slice (commit 582d1cf, `src/arbfree_vol/ssvi/term_structure.py`). Per-slice rho is fully free (tanh-reparametrised per slice, no cross-slice functional form). Slices that fit within the hard constraints are arbitrage-free BY CONSTRUCTION; slices that fall back to the unconstrained per-slice fit are NOT (see `RepairReport.fallback_slices` / `repair_infeasible` and Issue #15) — the grid-based `detect_svi_surface` then reports those violations as remaining_violations rather than being only a redundant regression check. The **SABR** path is an empirical comparison parametrisation: it fits cubic B-spline term structures on alpha(t)/nu(t)/rho(t) across expiries with a cross-slice calendar-arb SOFT penalty, with B-spline coefficients reparametrised at the control-point level (scaled tanh for rho, exp+floor for alpha/nu) to stay in-range between knots by the convex-hull property (`src/arbfree_vol/sabr/term_structure.py`). Calendar-arb verification is grid-based and EMPIRICAL -- not a closed-form guarantee; dynamic SABR is a not-implemented research extension. All three paths report remaining violations honestly via `detect_svi_surface`; `RepairReport.repair_infeasible` is set true if the eSSVI hard constraints cannot be satisfied.

## API Layer

The FastAPI layer can expose endpoints such as:

```text
GET /health
POST /chains/ingest
GET /chains/{chain_id}
POST /chains/{chain_id}/clean
POST /chains/{chain_id}/implied-vols
POST /chains/{chain_id}/arbitrage-check
POST /chains/{chain_id}/fit-svi
POST /chains/{chain_id}/repair
GET /surfaces/{surface_id}
GET /surfaces/{surface_id}/violations
```

The API should be added after the core quant engine is tested.

## Dashboard Ideas

The dashboard should come later, after the core math engine works.

Useful views:

- Raw option chain table.
- Raw implied volatility smile by expiry.
- Fitted SVI smile by expiry.
- Full volatility surface.
- Arbitrage violation heatmap.
- Raw vs repaired surface comparison.
- Quote rejection report.
- SVI parameter table.
- Pricing and Greeks calculator.

The dashboard should support exploration, not just decoration.

## Testing Strategy

Tests should be part of the project from the beginning.

Important tests:

- Black-Scholes known-value tests.
- Put-call parity tests.
- Greeks sanity checks.
- IV solver round-trip tests.
- IV solver invalid-input tests.
- Strike monotonicity violation tests.
- Butterfly arbitrage detection tests.
- Calendar arbitrage detection tests.
- SVI function shape tests.
- SVI calibration smoke tests.
- Repair engine regression tests.

The project should avoid false confidence. Numerical tolerances should be explicit.

## Documentation Plan

The docs should be serious enough for an interviewer or quant-minded reader.

Recommended docs:

### `docs/architecture.md`

Explain:

- System components.
- Data flow.
- Module boundaries.
- Storage design.
- API design.
- Why the project is not notebook-only.

### `docs/no_arbitrage_conditions.md`

Explain:

- No-arbitrage bounds.
- Monotonicity across strikes.
- Convexity across strikes.
- Calendar arbitrage.
- Put-call parity.
- Breeden-Litzenberger density relationship.

### `docs/svi.md`

Explain:

- Total variance.
- Log-moneyness.
- SVI parameterization.
- Calibration objective.
- Parameter constraints.
- Known limitations.

### `docs/data_cleaning.md`

Explain:

- Why option data is noisy.
- Quote rejection rules.
- Handling illiquid contracts.
- Handling wide spreads.
- Preserving audit trails for rejected quotes.

## Suggested Roadmap

### Phase 1: Core Package

Goal: build the mathematical foundation.

- Scaffold Python package.
- Add `pyproject.toml`.
- Add typed option quote models.
- Implement Black-Scholes pricing.
- Implement Greeks.
- Implement implied volatility solver.
- Add unit tests.

### Phase 2: Arbitrage Detection

Goal: detect bad surfaces before fitting.

- Implement quote cleaning.
- Implement monotonicity checks.
- Implement butterfly convexity checks.
- Implement calendar checks.
- Implement put-call parity checks.
- Add violation reporting models.

### Phase 3: SVI Fitting

Goal: fit smooth volatility smiles.

- Implement raw SVI function.
- Convert strikes to log-moneyness.
- Convert IV to total variance.
- Fit one expiry slice.
- Fit all expiry slices.
- Add calibration diagnostics.

### Phase 4: Repair Engine

Goal: transform noisy quotes into a usable surface.

- Remove impossible quotes.
- Refit after quote rejection.
- Add penalties for arbitrage violations.
- Enforce maturity consistency where practical.
- Output repaired surface and diagnostics.

### Phase 5: Storage And API

Goal: make the system feel production-grade.

- Add DuckDB storage.
- Store raw chains, cleaned chains, fitted surfaces, and repair reports.
- Add FastAPI endpoints.
- Add request and response schemas.

### Phase 6: Visualization

Goal: make the work explorable.

- Add surface plots.
- Add smile-by-expiry plots.
- Add raw vs repaired comparisons.
- Add violation heatmaps.
- Optionally build a TypeScript dashboard.

### Phase 7: Advanced Extensions

Optional, once the core is strong:

- SABR fitting. **(implemented — `repair(use_sabr=True)`, `src/arbfree_vol/sabr/`; term-structure B-spline calibration in `sabr/term_structure.py`)**
- eSSVI arbitrage-free-by-construction calibration (hard-constrained slices; fallback slices excepted — see `repair_infeasible`/Issue #15). **(implemented — `ssvi/term_structure.py:fit_ssvi_surface_sequential`, Hendriks & Martini 2019 Prop 3.1; commit 582d1cf)**
- Local volatility extraction. **(implemented — `pricing/local_vol.py`, Dupire)**
- Heston calibration.
- American option adjustments.
- Rust or C++ acceleration for IV solving.
- Streaming option chain updates.
- Distributed calibration workers.

## What Makes This Project Stand Out

This project is rare because it does not just apply a model to financial data. It confronts the actual structure of option markets:

- Market quotes are noisy.
- Implied volatility is not directly observed.
- Surfaces can violate arbitrage.
- Calibration can be unstable.
- Clean architecture matters if the system is meant to be reused.

A strong implementation will show that the builder understands both the math and the engineering.

## Resume Bullet

Built an arbitrage-free implied volatility surface engine that ingests option chains, solves implied volatilities, detects calendar and butterfly arbitrage, fits SVI total variance slices, and repairs noisy market surfaces under no-arbitrage constraints.

## Interview Talking Points

Be ready to explain:

- Why raw implied volatility surfaces can violate arbitrage.
- Difference between price interpolation and total variance interpolation.
- Why total variance matters.
- How implied volatility inversion works.
- Why deep OTM or near-expiry options are numerically difficult.
- What butterfly arbitrage means.
- What calendar arbitrage means.
- Why SVI is useful.
- How you designed the system so it is more than a notebook.
- How you tested numerical code.
- What tradeoffs you made in the repair engine.

## Non-Goals

This project should not become:

- A generic stock prediction model.
- A one-file notebook.
- A dashboard without a real quant engine.
- A thin wrapper around an external library.
- A project that hides failed calibration cases.

It is better to have a smaller, correct, well-tested engine than a huge but fragile demo.

## Initial Prompt For A New Build Session

```text
I want to build a serious quant finance project called `arbfree-vol-surface`: an arbitrage-free implied volatility surface engine.

Context:
I am a math major interested in finance, stochastic calculus, and strong system design. I do not want a generic resume project. I want this to be rigorous, difficult, and impressive enough to discuss in quant/dev interviews.

Project goal:
Build a production-style system that ingests option chain data, computes implied volatilities, detects static arbitrage violations, fits SVI-style volatility surfaces, and repairs noisy surfaces under no-arbitrage constraints.

Start Python-first. Build this as a clean package, not as a notebook-only project. Use tests from the beginning.

First task:
Inspect the workspace, propose the initial architecture, then scaffold the first version of the project with package structure, README, docs stubs, and initial tests for Black-Scholes and implied volatility.
```

