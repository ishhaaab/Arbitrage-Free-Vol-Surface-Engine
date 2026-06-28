# arbfree-vol-surface

An arbitrage-free implied volatility surface engine built in Python.

This project is a production-style quant finance system for pricing options, computing implied volatilities, detecting static arbitrage violations, and eventually fitting and repairing volatility surfaces.

## Project Goal

Raw option market data is noisy. Bid/ask spreads, stale quotes, illiquid contracts, crossed markets, and sparse expiries can all create implied volatility surfaces that violate no-arbitrage conditions.

The goal of this project is to build a clean, tested Python engine that can:

- Represent option contracts with typed models
- Price European calls and puts using Black-Scholes
- Compute Greeks
- Solve implied volatility from market prices
- Detect static arbitrage violations
- Fit SVI-style volatility smiles and surfaces
- Repair noisy surfaces under no-arbitrage constraints

The dashboard and API will come later. The first priority is correctness, numerical stability, and clean architecture.

## Current Scope

The project currently focuses on the Python-only quant core:

- Typed option models
- Black-Scholes pricing
- Greeks
- Implied volatility solver
- Unit tests

## Planned Features

### Core Pricing

- European call and put pricing
- Continuous dividend yield support
- Delta, gamma, vega, theta, and rho
- Numerical tests against known Black-Scholes values

### Implied Volatility

- Robust implied volatility solver
- Brent-style root finding
- Invalid price handling
- Round-trip pricing tests

### Static Arbitrage Detection

- Strike monotonicity checks
- Butterfly convexity checks
- Calendar arbitrage checks
- Put-call parity checks
- Negative density diagnostics

### SVI Surface Fitting

- SVI total variance parameterization
- Per-expiry smile fitting
- Calibration diagnostics
- Parameter constraints

### Repair Engine

- Quote rejection and filtering
- Arbitrage-aware surface repair
- Clean implied volatility grid generation
- Repair diagnostics and quality metrics

## Tech Stack

- Python
- NumPy
- SciPy
- Pydantic
- Pytest
- Pylance (Pyright-based type checking)

Future additions may include:

- FastAPI
- DuckDB
- Plotly
- A TypeScript dashboard

## Project Structure

```text
arbfree-vol-surface/
  pyproject.toml
  README.md
  src/
    arbfree_vol/
      models/
        option.py
      pricing/
        black_scholes.py
        greeks.py
        implied_vol.py
  tests/
    test_black_scholes.py
    test_implied_vol.py