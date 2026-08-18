# SVI, SSVI and eSSVI — Volatility Parameterizations

> Notation follows Gatheral (2004) for raw SVI and Gatheral & Jacquier (2014) for SSVI/eSSVI. Code variable names match the papers: `a, b, rho, m, sigma` for raw SVI; `theta, rho, psi, eta, gamma` for eSSVI. All line references are HEAD.

---

## 1. Intuition — Why SVI, Why Total Variance

Raw implied volatility `σ_imp(K,T)` mixes time-scaling with smile shape: calendar arbitrage (`∂w/∂T < 0`) is invisible in `σ` alone, interpolation in `σ` or price violates convexity, and a single bad quote can bend the whole surface. **Total variance** linearizes the arb-free structure:

* `w(k,T)` is the natural state variable — calendar arb is simply `w(k,T)` non-decreasing in `T`.
* SVI/SSVI are closed forms for `w(k)`, not `σ(k)`.
* Linear interpolation in `w` between expiries gives `σ = sqrt(w/T)` for free — this is exactly what `surface/interpolate.py:110 total_variance_at` / `202 iv_at` do.

SVI compresses a noisy smile (dozens of strikes) into 5 interpretable parameters with controlled asymptotics (linear wings, hyperbolic center). SSVI ties the slices together with 3 per-slice parameters under a guaranteed arb-free surface; eSSVI adds a one-parameter wing-decay law so the whole term structure shares a global `η, γ`.

**What breaks without it:** interpolating `C(K)` or `σ(K)` directly produces negative butterfly spreads (`∂²C/∂K² < 0` ⇔ negative risk-neutral density) and calendar inversions. SVI's `g(k) ≥ 0` condition and SSVI's Gatheral–Jacquier + Hendriks & Martini conditions are the parameterized guards against exactly those failures.

---

## 2. Core Definitions

```
w(k,T) = σ_imp(k,T)² · T            total variance
k      = ln(K / F)                  log-moneyness (forward-moneyness)
F      = S · exp((r - q)·T)         forward price
```

| Symbol | Meaning | Source |
|---|---|---|
| `S` | spot (`VolSurface.spot`) | `models/surface.py` |
| `K` | strike (`Quote.strike`) | `models/option.py` |
| `T` | time to expiry in years, ACT/365 `days/365.0` | `ingestion/loader.py:32`, `ingestion/yfinance.py:143`, `ExpirySlice.expiry_time` |
| `r, q` | risk-free rate, dividend yield — per-slice override preferred via `get_r`/`get_q` (`models/surface.py:31,41`), surface-level fallback | `repair/fwd_curve.py:estimate_forward_curve` populates per-slice `r` |
| `F` | forward | `svi/data.py:_forward_price` (same formula in `repair/fwd_curve.py`) |
| `w` | market total variance | `variance.py:slice_total_variance` — inverts `pricing/implied_vol.py:implied_vol` to `σ`, forms `w = σ²·T`; quotes with `None` IV dropped; call+put at same strike averaged |

Day-count is fixed ACT/365; changing it means changing the `365.0` divisor in both ingestion paths.

---

## 3. Raw SVI

Gatheral (2004), `svi/model.py:40 svi_total_variance`:

```
w(k) = a + b · ( ρ·(k - m) + sqrt((k - m)² + σ²) )
```

| Param | Role | Constraint | Code |
|---|---|---|---|
| `a` | vertical level (ATM variance anchor) | `a ∈ ℝ` but `w_min ≥ 0` required | `SVIParams.a` |
| `b` | slope / wing steepness | `b ≥ 0` | `Field(ge=0)` |
| `ρ` (rho) | skew orientation (sign = put/call wing tilt) | `-1 < ρ < 1` | `Field(gt=-1, lt=1)`, optimizer clips to `±0.999` |
| `m` | horizontal shift of smile minimum | `m ∈ ℝ` | `SVIParams.m` |
| `σ` (sigma) | curvature around the minimum (small σ = tight U, large σ = flat) | `σ > 0` | `Field(gt=0)`, optimizer `lb=1e-6` |

Implicit: `w(k) > 0 ∀ k` ⇔ `w_min = a + b·σ·√(1-ρ²) ≥ 0` (`svi/model.py:18`, `arbitrage/svi_detect.py:8 min_total_variance` / `:18 _check_min_variance`).

**Density condition** `svi/model.py:29 svi_g` (with `svi_core` at `svi/model.py:18` returning `(w, w', w'')`):

```
u   = k - m,  R = sqrt(u² + σ²)
w   = a + b·(ρ·u + R)
w'  = b·(ρ + u/R)
w'' = b·σ² / R³
g(k)= (1 - k·w'/(2w))² - (w'²/4)·(1/w + 1/4) + w''/2
```

`g(k) ≥ 0 ∀ k` ⇔ no butterfly arbitrage (non-negative risk-neutral density `f_Q ≥ 0` via Breeden–Litzenberger `∂²C/∂K² = e^{-rT} f_Q`). If `w ≤ 0`, `svi_g` returns `-inf`. Detection uses `arbitrage/svi_detect.py:37` (121-pt coarse scan over `[-3,3]` + Brent refinement in ±3 steps — fixes the `minimize_scalar(bounded)` spike trap of `docs/issues.md` #1).

---

## 4. SSVI

Gatheral & Jacquier (2014), `ssvi/model.py:60 ssvi_w`:

```
w(k) = (θ/2) · ( 1 + ρ·ψ·k + sqrt((ψ·k + ρ)² + (1 - ρ²)) )
```

| Param | Role | Constraint | Code |
|---|---|---|---|
| `θ` (theta) | ATM total variance `σ_ATM²·T` | `θ > 0` | `SSVIParams.theta  Field(gt=0)` |
| `ρ` (rho) | spot-vol correlation | `-1 < ρ < 1` | `SSVIParams.rho` |
| `ψ` (psi) | angle / wing slope at this `θ` | `ψ > 0` | `SSVIParams.psi  Field(gt=0)` |

Helpers: `ssvi_dw_dk` (`ssvi/model.py:74`), `ssvi_d2w_dk2` (`:81`). Per-slice type `SSVIParams(theta, rho, psi)` (`ssvi/model.py:31`).

**Gatheral–Jacquier butterfly bounds** `ssvi/model.py:88 gatheral_jacquier_condition(theta, rho, psi, strict=False)` — Theorem 4.2, sufficient per-slice no-butterfly:

```
θ·ψ·(1+|ρ|)  <  4    (STRICT — condition 1)
θ·ψ²·(1+|ρ|) ≤  4    (non-strict — condition 2)
residual = min(4 - θψ(1+|ρ|), 4 - θψ²(1+|ρ|))  ≥ 0  ⇔  arb-free
```

With `strict=True` the exact boundary of condition 1 (within `_GJ_STRICT_EPS = 1e-9` at `ssvi/model.py:28`) is nudged negative so equality is never reported safe. Production optimizer imports the same constant as `_GJ_CONDITION1_STRICT_EPS` (`ssvi/term_structure.py:68` via `ssvi/_butterfly.py`) — diagnostic and calibration cannot diverge. Also exposed are `essvi_params_in_bounds` / alias `essvi_arb_safe` (`ssvi/model.py:148`) which checks only `0 ≤ γ ≤ 1, η > 0` and is **not** an arb-free guarantee (`docs/issues.md` #9).

---

## 5. eSSVI — Power-Law Wing

`ssvi/model.py:53 essvi_psi` / `:69 essvi_w`:

```
ψ(θ) = η / θ^γ
w_eSSVI(k; θ, ρ, η, γ) = ssvi_w(k, θ, ρ, η/θ^γ)
```

| Param | Role | Constraint |
|---|---|---|
| `η` (eta) | power-law coefficient | `η > 0` (`eSSVISurfaceParams  Field(gt=0)`) |
| `γ` (gamma) | power-law exponent — wing flattening with maturity | `0 ≤ γ ≤ 1` (`Field(ge=0, le=1)`) |

`γ = 0` → `ψ = η` (SSVI with flat wing); `γ = 1` → `ψ ∝ 1/θ` (strong decay, flatter long-dated wings). The interval `0 ≤ γ ≤ 1` is the necessary structural bound; full arb-free needs the per-slice GJ check plus the term-structure conditions below. `essvi_psi` raises `ValueError` if `θ ≤ 0`. Type `eSSVISurfaceParams(eta, gamma)` at `ssvi/model.py:43`.

> In this codebase the sequential fit does **not** share a global `(η,γ)` — it fits per-slice `ψ_i` with `χ_i = θ_i·ψ_i` constrained by H&M (next section). The `(η,γ)` form is the diagnostic/legacy surface description and the `essvi_*` helpers.

---

## 6. Calibration

All fits minimize squared error **in total variance**:

```
min_p  Σ_i  ( w_model(k_i; p) - w_market(k_i) )²     where w_market = σ_imp²·T
```

### 6.1 Unconstrained — `svi/calibration.py:128 calibrate`

* `scipy.optimize.least_squares` with bounds `[(-∞, 0, -0.999, -∞, 1e-6), (∞, ∞, 0.999, ∞, ∞)]` — only `b, ρ, σ` bounded; `a, m` free.
* Seed `x0 = [min(w), 0.1, -0.5, 0.0, 0.1]`.
* No arb penalty. Preserved for back-compat, tests, and as the warm-start source for the constrained path.
* Optional `max_nfev` threads to `least_squares`.

### 6.2 Constrained — `svi/calibration.py:154 calibrate_constrained`

Augments the residual vector with **soft penalties** so a feasible fit collapses to the unconstrained objective (`svi/calibration.py:38 _build_residuals`):

```python
sqrt_pen = sqrt(arb_penalty)          # default 100.0
residuals = [
    w_model(k_i) - w_i,                           # data fit
    sqrt_pen * sqrt(max(-g(k_j), 0)),             # butterfly  — each k_j on grid
    sqrt_pen * sqrt(max(-w_min, 0)),              # min-variance
    sqrt_pen * sqrt(max(w_prev(k_j)-w(k_j), 0)),  # calendar — only if prev_slice given
]
# _penalty_term at calibration.py:33 is sqrt_pen * sqrt(max(value,0))
```

* `k_grid = linspace(k_min, k_max, n_k)` defaults `k_min=-3.0, k_max=3.0, n_k=121`.
* `prev_slice: SVIParams | None` — when supplied adds the calendar penalty `w(k) ≥ w_prev(k)` on the same grid. **Only the raw-SVI repair path threads this**, via `_fit_slice` in `repair/engine.py` in ascending-expiry order. eSSVI/SABR do not use it — they have dedicated term-structure fitters.
* **Multi-start** (`calibration.py:91 _run_multistart`): two seeds, lowest-cost *successful* finite-cost result wins; `ValueError` (non-finite start, e.g. live SPY extreme `b≈30, ρ≈0.997`) and unsuccessful runs are skipped:
  1. Fixed default `x0 = [min(w), 0.1, -0.5, 0.0, 0.1]`
  2. Warm start from `calibrate(points, max_nfev=_WARM_START_MAX_NFEV)` where `_WARM_START_MAX_NFEV = 150` (`calibration.py:25`). Tuned so clean/synthetic data (9–116 nfev) still converges while large real slices (545 strikes, ~10.6k nfev for full convergence) fail fast instead of burning ~1.1 s per slice. Saves ~8.5 s → ~2.6 s across the 7-slice SPX fixture.

### 6.3 Optimizer Choice — `least_squares` vs `trust-constr`

| Path | Optimizer | Constraints | File |
|---|---|---|---|
| Raw SVI `calibrate` / `calibrate_constrained` | `scipy.optimize.least_squares` (TRF) | bounds + soft penalty residuals | `svi/calibration.py` |
| eSSVI sequential `_fit_slice` | `scipy.optimize.minimize` **trust-constr** → **SLSQP** retry | hard `NonlinearConstraint` (H&M + GJ) | `ssvi/_constraints.py:99 _constrained_minimize`, `ssvi/term_structure.py:192 _fit_slice` |

`_constrained_minimize` runs `trust-constr` (`gtol=1e-10, maxiter=500`) first; only statuses 1/2 (= `success True`) count. Status 4 ("minimize successful but constraints not satisfied") and 0/3 are failures. On failure it retries **SLSQP** from `result.x` (`ftol=1e-12, maxiter=1000`); SLSQP mode 0 only is success — modes 1/2/3 would silently certify a non-converged slice as arb-free. Both failing → `RuntimeError` routes the slice to fallback bookkeeping. `minimize` is injected as `minimize_fn` so tests can patch `term_structure.minimize` to script optimizer statuses.

Bounds for the eSSVI slice are in unconstrained space `(θ, u=arctanh ρ, v=log ψ)`: `Bounds(lb=[1e-6,-6,log(1e-8)], ub=[10,6,log(20)])`, with `ρ = tanh u`, `ψ = exp v`. Initial guess `_initial_guess` (`term_structure.py:147`) runs an inner `least_squares` seed then bumps `θ, ψ` to satisfy `θ ≥ θ_prev+eps_theta`, `χ ≥ χ_prev+eps_chi`.

---

## 7. Hard-Constrained eSSVI Sequential Fit (Hendriks & Martini Prop 3.1)

`ssvi/term_structure.py:388 fit_ssvi_surface_sequential` — slices sorted by `T₁ < T₂ < … < Tₙ`, each inherits constraints from its predecessor. Per-slice `ρ` stays fully free (tanh-reparameterised, no cross-slice functional form). Reference: Hendriks & Martini (2019) Prop 3.1; Corbetta et al. (2019) §2.2–2.3 (arXiv:1804.04924).

For `χ_i = θ_i·ψ_i`:

```
(a) θ₁ ≤ θ₂ ≤ … ≤ θₙ                                              (ATM variance)
(b) χ₁ ≤ χ₂ ≤ … ≤ χₙ                                              (wing magnitude)
(c) |(ρ_{i+1}·χ_{i+1} - ρ_i·χ_i) / (χ_{i+1} - χ_i)| ≤ 1   ∀ i       (slope)
    + both GJ butterfly bounds per slice
```

Implemented as hard `NonlinearConstraint`s in `ssvi/_constraints.py:27 _hard_constraints` (re-exported via `term_structure.py:68`):

* **Butterfly** — 4 residuals `≥ 0` via `_butterfly_constraints` (`ssvi/_butterfly.py`, split `(1+ρ)` / `(1-ρ)` form), `NonlinearConstraint(_bf_con, 0, inf)`.
* **(a)** `θ - θ_prev ≥ eps_theta` (`_EPS_THETA` from `ssvi/_hm_margin.py`).
* **(b)** `χ - χ_prev ≥ eps_chi` (`_EPS_CHI`).
* **(c)** ratio as two constraints `±(ρ·χ - ρ_prev·χ_prev)/(χ-χ_prev) ∈ [-1,1]` (`_ratio_upper` / `_ratio_lower`, denominator floored at `eps_chi`).

Fallback is honest: on hard-fit `RuntimeError` (or degenerate corner — see below) the slice falls back to unconstrained `fit_ssvi_slice` (`ssvi/calibration.py`). That fallback is **not** arb-free — listed in `RepairReport.fallback_slices` / `failed_slices` (`repair/report.py`), `repair_infeasible=True`, surfaced by `detect_svi_surface` as `remaining_violations`. The fallback result is **not** used as `prev` for the next slice — `last_valid_prev` stays the last hard-constrained success (`term_structure.py:441`).

**Degenerate boundary corner** (`term_structure.py:261 _hard_fit_is_degenerate_corner`) — a hard fit pinned within `10×eps` of the H&M boundary with anomalously bad RMSE (`hard_rmse > max(_HM_RMSE_RATIO_MAX * unconstrained_rmse, _HM_RMSE_FLOOR)` from `ssvi/_hm_margin.py`) is treated as `RuntimeError` and routed to fallback. Without a baseline, boundary proximity alone flags. First slice (`prev=None`) is never flagged. This catches the m66/mutmut_66 pattern (`θ_delta≈1e-9, χ_delta≈1e-6, ratio≈0.9998, hard RMSE 0.05 vs unconstrained 1.6e-11`). The boundary-pinned-escape class on the GJ condition-2 boundary (RMSE 0.07–0.32) is a separate OPEN quality issue (`docs/issues.md` § OPEN).

Post-fit `verify_hm_condition` / `verify_ssvi_calendar_free` (`ssvi/_hm_verify.py`, re-exported via `term_structure.py:81`) gate the result — `fit_ssvi_surface_sequential` warns if `verify_hm_condition` fails, and the repair engine's `detect_svi_surface` is the load-bearing grid verification. Verification grids: native SSVI post-fit `verify_ssvi_calendar_free` over `linspace(-3,3,241)`; see §10 for the narrower SVI calendar grid.

**SABR term structure** (`sabr/term_structure.py:fit_sabr_term_structure`) is the contrast: cubic B-spline term structures on `α(t), ν(t), ρ(t)` (β fixed hint) with coefficient-level reparameterisation (scaled tanh for `ρ`, `exp+floor` for `α/ν`, convex-hull property keeps the curve in-range between knots) and a **soft** calendar penalty via `least_squares` — empirical, not arb-free by construction.

---

## 8. Mapping Between Parameterizations

### SSVI → raw SVI — exact

`ssvi/model.py:168 to_raw_svi_params(theta, rho, psi) -> (a,b,rho,m,sigma)`:

```
b     = θ·ψ / 2
m     = -ρ / ψ
σ     = sqrt(1 - ρ²) / ψ
a     = (θ/2)·(1 - ρ²)
ρ     passed through unchanged          (raises ValueError if ψ ≤ 0)
```

Exact for all `k`. Lets the eSSVI surface reuse the raw-SVI pipeline (detection, `surface/interpolate.py`, viz) unchanged.

### SABR → raw SVI — least-squares approximation

`sabr/model.py:190 to_raw_svi_params(SABRParams, forward, expiry_time, k_grid=None)` — samples `w_SABR(k) = sabr_total_variance(k, F, T, α,β,ρ,ν)` on `_DEFAULT_K_GRID` (200 pts: 50+100+50, centered on `[-3,3]` with 100 inside `[-1,1]`; `sabr/model.py:30`) then `least_squares` to raw SVI (`max_nfev=50000`). Seed `x0=[w0/2, 0.2, 0.0, 0.0, 0.3]` where `w0 = w_SABR(0)`. Not exact — approximation RMSE not exposed on the return (`docs/issues.md` #12; `FittedSlice.rmse` in the repair path is SABR-to-data, not SVI-to-SABR). Center-weighted grid mitigated a prior ~0.021 vol max reprice error → ~0.0175 vol.

SABR IV itself is Hagan et al. (2002) Eq. 2.17a asymptotic `sabr_implied_vol(k,F,T,α,β,ρ,ν)` (`sabr/model.py:111`) with ATM limit for `|k| ≤ 1e-8`; `sabr_total_variance = σ_B²·T`.

---

## 9. Model Comparison

| Dimension | Raw SVI | SSVI | eSSVI (this codebase) | SABR |
|---|---|---|---|---|
| **Params per slice** | 5: `a,b,ρ,m,σ` | 3: `θ,ρ,ψ` | 3 per slice `θ,ρ,ψ` with `ψ=η/θ^γ` globally (here: per-slice `ψ_i` with `χ=θψ` H&M-constrained) | 4 per slice `α,β,ρ,ν` (β fixed hint) |
| **Arb-free guarantee** | None by construction — needs penalty or detection | Per-slice GJ sufficient for no butterfly; needs H&M for calendar | **Hard-constrained slices**: GJ + H&M Prop 3.1 as hard optimizer constraints + grid calendar gate — discrete certificate (grid `[-3,3]`), not global analytic (see §10). **Fallback slices**: not certified (`repair_infeasible=True`) | None by construction — B-spline + soft penalty, empirical grid check only |
| **Cross-slice coupling** | Soft calendar penalty threaded via `prev_slice` (raw SVI path only) | Uncoupled unless H&M added | Sequential hard constraints (a)(b)(c); fallback `prev` frozen | B-spline `α(t),ν(t),ρ(t)` + soft penalty |
| **Typical use** | Interpretable single-slice smile; research / diagnostics | Single-slice arb-free smile | Production surface — primary certified path (`repair(use_ssvi=True)`) | Comparison / rates parametrization; vol-of-vol intuition |
| **Calibrator** | `least_squares` + soft `arb_penalty` | `least_squares` per slice | `minimize trust-constr → SLSQP` sequential | `least_squares` per slice + B-spline `least_squares` |
| **Pipeline** | `repair(use_ssvi=False,use_sabr=False)` | via `to_raw_svi_params` adapter | `repair(use_ssvi=True)` → `fit_ssvi_surface_sequential` | `repair(use_sabr=True)` → `fit_sabr_term_structure`; `to_raw_svi_params` adapter |

All three adapters feed `FittedSurface` / `iv_at` / Greeks / Dupire unchanged.

---

## 10. Known Limitations

### Near-expiry steep wings — `docs/issues.md` #3

For `T < 0.05` (~18 d) the right wing (OTM calls, SPY) is steep enough that raw SVI fits visually but violates `g(k) < 0` at `k≈0.5–1.5`:

```
SVI butterfly arbitrage: g=-14.88 < 0 at k=0.2473
SVI butterfly arbitrage: g=-90.45 < 0 at k=1.8061
```

Model violations, not quote violations — raw SVI is not arb-free by construction. Mitigations: `calibrate_constrained(arb_penalty=…)`, eSSVI hard constraints, or pragmatically skip `T < 14d`. Reported honestly as `remaining_violations`.

### Calendar `k`-range narrower than butterfly — `docs/issues.md` #4

`svi_detect.py:80 _check_calendar` defaults `k_grid = linspace(-1.5,1.5,121)` while `_check_butterfly` scans `[-3.0,3.0]` and the native SSVI gate `verify_ssvi_calendar_free` uses `[-3,3]`×241. Wing-only calendar violations with `|k| > 1.5` are missed at defaults. Pass an explicit wider `k_grid` when auditing wings.

### Theta dips / fallback slices — `docs/issues.md` #15

On live SPY ~4–7 of ~20 eSSVI slices fall back (up to 21/40 on SPX raw in snapshots) because ATM total variance `θ = σ²T` dips non-monotonically in the short end:

```
T=0.0658  θ=0.003609  (predecessor)
T=0.0849  θ=0.002563  (FALLBACK — dips 29%)
T=0.0932  θ=0.003632  (recovers)
```

H&M (a) requires `θ` non-decreasing; when data wants a dip the optimizer reports "Positive directional derivative for linesearch" (saddle/boundary). Warm-start from the unconstrained solution fails 100% (it violates (a) itself); random restarts converge to degenerate copies of `prev` with 1.1–3.6× worse RMSE. Root causes: event risk concentration (earnings/FOMC), microstructure noise, `θ = σ²T` amplifying ATM vol noise. H&M is sufficient, not necessary — a dip can still be arb-free with skew adjustment, but this parametrization cannot represent it. Fallbacks are `RepairReport.fallback_slices` with `repair_infeasible=True`; Dupire stencil NaN-propagation (`pricing/local_vol.py`) grays out fallback-adjacent rows. Counts are snapshot-in-time — expiries roll daily; determinism was verified byte-identical on `tests/fixtures/spx_sample.json`.

### Grid-based certificate — `docs/issues.md` § eSSVI calendar certificate is grid-based

Hard-constrained eSSVI satisfies the **implemented** GJ conditions and the **grid-based** calendar check — a discrete certificate. The exact H&M Prop 3.1/3.5 sufficient disjunction is OPEN (paywalled primary). Counterexample still reproducible at HEAD (FIX-6 probe):

```
p1 = SSVIParams(theta=0.01495, rho=-0.65485, psi=0.11492)
p2 = SSVIParams(theta=0.05749, rho=-0.88305, psi=2.52265)
GJ residuals all ≥0, χ 0.00171→0.14504, verify_hm_condition True
w2-w1 at k=0.68 = -0.001528  ← calendar violation between grid points
```

At HEAD `verify_ssvi_calendar_free` over `[-3,3]`×241 catches this pair (`min w2-w1 = -0.00153` at `k≈0.675` → `repair_infeasible=True`), but violations strictly between points or beyond `[-3,3]` remain uncertified. The underlying Prop 3.5 disjunction is not yet in `verify_hm_condition`.

### Degenerate boundary escape — `docs/issues.md` § OPEN hard-fit quality escape

A hard fit pinned on the GJ condition-2 boundary can pass all grid checks yet fit poorly (RMSE 0.07–0.32, vs `_hard_fit_is_degenerate_corner`'s m66 class). Reported honestly with `repair_infeasible=False` but poor `FittedSlice.rmse` — a fit-quality issue, not a silent certification bug.

---

## 11. Interview Cheat Sheet

| Question | One-liner |
|---|---|
| Why `w = σ²T` not `σ`? | `w` is the arb-free state variable; `σ` mixes time scaling. Calendar arb is `w(k,T) ↓`, invisible in `σ` or `C`. |
| Why `k = ln(K/F)` not `ln(K/S)`? | Forward-moneyness removes drift; smile is stationary in `k`. `F` via `S·exp((r-q)T)` per slice. |
| What does each raw SVI param do? | `a` level, `b` wing steepness, `ρ` skew sign, `m` smile shift, `σ` curvature. |
| `g(k)` spike trap? | `g(k)` spikes to +300 at `k=m` for tiny `σ`; bare Brent brackets the wrong side. Fix: 121-pt scan + local Brent ±3 steps. |
| Calendar narrow grid? | `svi_detect` calendar `[-1.5,1.5]` misses wing violations; use `[-3,3]` or `verify_ssvi_calendar_free`. |
| What does GJ guarantee? | Sufficient (not necessary) per-slice no-butterfly when both `θψ(1+|ρ|)<4` (strict) and `θψ²(1+|ρ|)≤4`. |
| H&M Prop 3.1 sufficient or necessary? | Sufficient for no calendar spread; a `θ` dip can still be arb-free with skew adjustment — hence fallback pathology. |
| `least_squares` vs `trust-constr`? | Raw SVI soft penalties → `least_squares`; eSSVI hard inequalities → `trust-constr` + SLSQP retry. |
| Warm-start trick? | Constrained multi-start: default seed + capped unconstrained `least_squares(max_nfev=150)` seed; caps large-slice burn. |
| `prev_slice` for what? | Raw SVI only — threads `w_prev(k)` calendar penalty in ascending `T` order (`repair/engine.py:_fit_slice`). eSSVI/SABR have term-structure fitters. |
| `to_raw_svi_params` exact? | SSVI→SVI exact (`b=θψ/2, m=-ρ/ψ, σ=√(1-ρ²)/ψ, a=θ(1-ρ²)/2`); SABR→SVI least-squares approximation (~0.0175 vol max reprice). |
| `repair_infeasible`? | `True` when any eSSVI hard fit failed or a grid gate tripped — surface not fully certified even if `remaining_violations` empty on a narrow grid. Fallback slices listed in `RepairReport.fallback_slices`. |
| Why median not mean for forwards? | One bad quote corrupts the mean; median parity-implied `F` is robust (`repair/fwd_curve.py`, `quote_detect.py`). |
| SABR vs eSSVI? | SABR empirical comparison (Hagan asymptotic, soft penalty, grid check); eSSVI primary certified surface (hard H&M+GJ, grid gate). |

---

## 12. References

* Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives.
* Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance* 14(1), 59–71. — `gatheral_jacquier_condition`, SSVI `w(k)`.
* Hendriks, S. & Martini, C. (2019). The Extended SSVI Volatility Surface. *J. Computational Finance* 22(5), Prop 3.1. — eSSVI wing `ψ=η/θ^γ`, term-structure conditions (a)(b)(c).
* Corbetta, J., Cohort, P., Laachir, I. & Martini, C. (2019). Robust calibration and arbitrage-free interpolation of SSVI slices. *arXiv:1804.04924*, §2.2–2.3. — sequential procedure.
* Hagan, P. S., Kumar, D., Lesniewski, A. S. & Woodward, D. E. (2002). Managing Smile Risk. *Wilmott Magazine*. Eq. 2.17a — `sabr/model.py:111 sabr_implied_vol`, `sabr_total_variance`.
* Breeden, D. & Litzenberger, R. (1978). Prices of State-Contingent Claims Implicit in Option Prices. *J. Business* 51(4). — `∂²C/∂K² = e^{-rT} f_Q`.
