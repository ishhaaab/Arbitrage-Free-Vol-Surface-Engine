# arbfree-vol-surface

[![CI](https://github.com/ishhaaab/Arbitrage-Free-Vol-Surface-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/ishhaaab/Arbitrage-Free-Vol-Surface-Engine/actions/workflows/ci.yml)

This project studies one question: can a noisy equity-index option chain be fitted closely while preserving static-arbitrage conditions on a useful numerical domain?

It loads a frozen SPX option snapshot, cleans quotes, estimates one forward per expiry from put-call parity, fits raw SVI as a baseline, fits sequentially constrained SSVI as the primary model, and checks the result on a fixed log-moneyness grid.

The project does not claim a globally arbitrage-free surface. Its certificate is discrete and limited to the domain printed in the report.

## Reproduce the study

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
python demo/run_study.py
```

The command reads only committed files. It writes:

```text
results/spx_2026-07-31/
  summary.json
  smiles.png
  surface.png
  constraints.png
```

Exit code `0` means the constrained fit passed the declared numerical checks. Exit code `2` means it did not. The output directory is generated and is not treated as source data.

The committed input is `data/spx_2026-07-31.csv`. Its source, timestamp, spot, flat continuously compounded rate, dividend yield, and ACT/365F convention are recorded in `data/spx_2026-07-31_metadata.json`.

Live Yahoo access is an optional collection step, not part of the study run:

```bash
pip install -e ".[yahoo]"
python scripts/fetch_yahoo_snapshot.py SPY --risk-free 0.04 --output data/spy_YYYY-MM-DD.csv
```

The collector requires the rate input instead of guessing one from another provider. Commit and review a refreshed snapshot before using it as reproducible evidence.

## Measured result

On the committed snapshot, using NumPy 2.5.2 and SciPy 1.18.1, the study produced:

| Quantity | Result |
|---|---:|
| Input quotes | 2,645 |
| Accepted quotes | 2,617 |
| Rejected by European price bounds | 28 |
| Raw SVI mean slice RMSE in total variance | 0.007595 |
| Constrained SSVI mean slice RMSE in total variance | 0.002726 |
| Certificate domain | `k in [-1.5, 1.5]` |
| Grid points | 241 |
| Tolerance | `1e-4` |
| Minimum total variance | `3.08e-5` |
| Minimum butterfly density `g(k)` | `0.2498` |
| Minimum adjacent calendar margin | `5.29e-5` |
| Constrained SSVI fallback or failed expiries | 0 |
| Numerically certified on that grid | yes |

Raw SVI fitted all seven expiries in this environment. Its failure list is still part of the report because the baseline is a comparison fit, not the certified output.

Optimizer results and runtime can change with NumPy and SciPy versions. `summary.json` records the installed versions for every run.

## Method

### Quote cleaning

`clean_quotes()` records one reason per rejected quote. It checks:

1. Finite, positive price
2. Present, positive, non-crossed bid and ask
3. Relative bid-ask spread
4. Minimum time to expiry
5. Discounted European lower and upper price bounds

It also applies a broad spot-log-moneyness cutoff. That cutoff defines the calibration sample. It is not evidence that a quote is economically invalid.

For maturity `T`, the cleaner uses the European bounds

```text
max(0, S exp(-qT) - K exp(-rT)) <= C <= S exp(-qT)
max(0, K exp(-rT) - S exp(-qT)) <= P <= K exp(-rT)
```

### Forward estimation

At each strike with both a call and a put, the code computes

```text
F_K = exp(rT) (C_K - P_K) + K
```

The expiry forward is the median of the positive strike-level estimates. If no pair exists, the code uses `S exp((r-q)T)` and reports the fixed carry assumptions in the dataset metadata. Before IV inversion, calibration derives the expiry dividend yield implied by that forward and the fixed risk-free rate. This keeps Black-Scholes inversion and `k = log(K/F)` on the same carry convention without mutating the loaded input. Parity is an input consistency diagnostic here, not a separate proof of an arbitrage-free fitted surface.

### Models

Raw SVI is the baseline. It shows that fit quality and static-arbitrage validity are different questions.

Sequentially constrained SSVI is the primary model. Each expiry has its own `(theta, rho, psi)` parameters. The optimizer enforces the implemented Gatheral-Jacquier butterfly bounds and adjacent-slice Hendriks-Martini parameter conditions. The implementation does not fit an eSSVI `eta/gamma` power law, so the project does not call this eSSVI.

### Numerical certificate

The post-fit verifier evaluates the raw-SVI-equivalent constrained slices on the same 241-point grid over `k in [-1.5, 1.5]`. It records:

1. Minimum total variance
2. Minimum Gatheral density condition `g(k)`
3. Minimum adjacent calendar margin `w(k,T_next) - w(k,T_prev)`

Certification also requires a non-empty fit and no fallback or failed constrained-SVI expiries. A violation between grid points or outside the configured interval may still exist. This is a numerical check, not a global analytic guarantee.

`build_fitted_surface()` rejects an uncertified report by default. Passing `allow_uncertified=True` is an explicit opt-in for investigation.

## Surface queries

```python
from arbfree_vol.surface import iv_at, total_variance_at

w = total_variance_at(surface, K=7500.0, T=0.25)
vol = iv_at(surface, K=7500.0, T=0.25)
```

Queries require positive strikes and maturities inside the fitted range. Maturity extrapolation is rejected. Strike queries use the SVI wings, including outside the observed strike sample.

## Repository scope

The maintained code covers Black-Scholes pricing, implied-volatility inversion, CSV ingestion, quote cleaning, parity forwards, raw SVI, constrained SSVI, numerical verification, surface interpolation, one report, and one demo.

SABR, OpenBB, FRED curves, custom market calendars, Greeks, Dupire local volatility, PCA dynamics, iterative quote repair, the general CLI, mutation-test infrastructure, and extra demos were removed. They added separate correctness obligations without strengthening this study.

## Tests

```bash
pytest tests -q
ruff check .
basedpyright
```

The focused suite covers known Black-Scholes values, implied-volatility round trips, discounted cleaning bounds, median forward estimation, SVI and SSVI recovery, constraint counterexamples, surface interpolation, failure reporting, and deterministic calibration on the frozen snapshot.

## References

- Gatheral, *The Volatility Surface: A Practitioner's Guide*, 2006.
- Gatheral and Jacquier, "Arbitrage-free SVI volatility surfaces," 2014.
- Hendriks and Martini, "The Extended SSVI Volatility Surface," 2019.
- Corbetta, Cohort, Laachir, and Martini, "Robust calibration and arbitrage-free interpolation of SSVI slices," 2019.
