# No-Arbitrage Conditions

> **One-line pitch:** A volatility surface is arbitrage-free when vanilla option prices at one observation time admit no static (no dynamic hedging) risk-free profit. Every condition below is a different projection of that requirement. Hard-constrained eSSVI slices satisfy the implemented Gatheral–Jacquier butterfly bounds and the **grid-based** calendar check; fallback slices and SABR do not. The certificate is discrete, not a global analytic guarantee — see §7.

## Why No-Arbitrage Matters for Vol Surfaces

Raw quotes are noisy: wide spreads, stale prints, illiquid wings, crossed markets. Naive interpolation of implied vol or price produces surfaces that price vanillas inconsistently at `t=0` — butterfly spreads with negative cost, calendar spreads that pay you to wait, call prices that rise with strike. Such a surface is unusable for:

* **Pricing** exotics (any price derived from it inherits the arb).
* **Risk** (Greeks and PnL on an arb surface are fictional).
* **Calibration** (optimizers chase inconsistent inputs).

Static arbitrage = arbitrage using only vanillas at a single time slice. It is weaker than dynamic arbitrage but is the correct bar for surface construction. All checks in `arbitrage/` are static; dynamic (model-dependent) arb is out of scope.

**Pipeline position:**

```
quotes → clean_quotes() → VolSurface → detect() / detect_with_forward()
                                     → repair() → detect_svi_surface()
                                     → RepairReport(remaining_violations, fallback_slices, repair_infeasible)
```

---

## Summary Table

| Condition | Math | `ViolationType` | Quote-level check | Curve-level check | Hard constraint | Tolerance |
|---|---|---|---|---|---|---|
| **Put-call parity** | `C-P = S e^{-qT} - K e^{-rT}` | `PARITY` | `quote_detect.py:61` | — | — | `max(Σ half-spreads, 0.05)` |
| **Strike monotonicity** | `C(K)` ↘, `P(K)` ↗ in `K` | `MONOTONICITY` | `quote_detect.py:201` | — | — | `1e-4` |
| **Butterfly / convexity** | `∂²C/∂K² ≥ 0 ⇔ g(k) ≥ 0` | `BUTTERFLY` | `quote_detect.py:226` (discrete) | `svi_detect.py:37` (`svi_g`) | GJ bounds (`ssvi/_butterfly.py`) | `1e-4` |
| **Negative variance** | `w_min = a+bσ√(1-ρ²) ≥ 0` | `NEGATIVE_VARIANCE` | — | `svi_detect.py:18` | implied by GJ | `1e-4` |
| **Calendar** | `w(k,T)` non-decreasing in `T` | `CALENDAR` | `calendar.py:22` / `quote_detect.py:4` | `svi_detect.py:80` | H&M Prop 3.1 (`ssvi/term_structure.py:388`) | `1e-4` |
| **Density** | `∂²C/∂K² = e^{-rT} f_Q(K)` | (via `BUTTERFLY`) | — | `svi/model.py:29` | — | — |

All arb tolerances are `1e-4` per `docs/AGENTS.md` → Numerical Rigor. Exceptions: cleaning intrinsic `1e-6`, IV Newton `1e-8`, GJ strict epsilon `1e-9`.

---

## 1. Put-Call Parity

### Math

For European options with same `K, T`:

```
F = S · exp((r - q)·T)                              (forward price)
C - P = S·e^{-qT} - K·e^{-rT} = e^{-rT}·(F - K)     (parity)
```

where `r = risk-free rate`, `q = dividend yield`, `F = forward`.

Violation ⇒ bad quote, wrong `r/q`, or American/borrow effect (not modelled).

### Intuition

Call minus put is a forward contract. If the market prices it differently from the forward, buy the cheap replication and sell the expensive one — immediate profit.

### Code Mapping

| Concept | Location |
|---|---|
| `ViolationType.PARITY` | `src/arbfree_vol/arbitrage/report.py:6` |
| Forward price `F = S·exp((r-q)T)` | `src/arbfree_vol/svi/data.py:_forward_price` and `src/arbfree_vol/forward.py` |
| Per-slice rate resolution | `src/arbfree_vol/models/surface.py:31` `get_r()` / `:41` `get_q()` — per-slice override preferred, falls back to surface-level |
| `r_eff` from estimated forward | `src/arbfree_vol/forward.py:populate_per_slice_r` — `r = log(F/S)/T + q` |
| Parity RHS (fallback mode) | `src/arbfree_vol/arbitrage/quote_detect.py:10` `_parity_rhs()` |
| Parity threshold | `src/arbfree_vol/arbitrage/quote_detect.py:25` `_parity_threshold()` |
| Parity check | `src/arbfree_vol/arbitrage/quote_detect.py:61` `_check_parity()` |
| Entry points | `quote_detect.py:304` `detect()` vs `quote_detect.py:315` `detect_with_forward()` |

### Detection Method

`_check_parity` operates per strike where both `C` and `P` exist (grouped by `quote_detect.py:44` `_group_by_strike`):

* **Forward-price mode** (`forward_price is not None`, preferred): `RHS = e^{-rT}(F-K)` where `F` is the **median** parity-implied forward for that expiry (see below).
* **Fallback mode** (`forward_price is None`): `RHS = S·e^{-qT} - K·e^{-rT}` via `_parity_rhs` using `get_r`/`get_q`.

Violation when `|C - P - RHS| > threshold`, flagged on both legs.

**Why `detect_with_forward` exists:** On live SPY data with default `r=0.05, q=0.0`, surface-level `r/q` flagged ~95% of strikes as parity violations (fixed in `docs/issues.md` #2). `detect_with_forward` runs `estimate_forward_curve()` as a pre-pass: per strike `F = e^{rT}(C-P)+K`, then **median** across strikes for that `T` (median, not mean — one outlier cannot corrupt it). Threaded into `_check_parity` and `_normalize_to_calls`. Parity rejection dropped to ~45%. Synthetic tests use `detect(surface)` where `r/q` is exact.

### Tolerance

```python
# quote_detect.py:25
if bid/ask present on both sides:
    threshold = max(half_spread_C + half_spread_P, 0.05)
else:
    threshold = 0.05
```

Sum of half-spreads = correct execution cost (buy one side at ask, sell the other at bid). `$0.05` floor calibrated for liquid US equities/ETFs (SPY, QQQ); index options (SPX/NDX) need `$0.10–$0.15`, illiquid names larger.

---

## 2. Strike Monotonicity

### Math

For fixed `T`:

```
K1 < K2  ⇒  C(K1) ≥ C(K2)    (calls non-increasing in strike)
K1 < K2  ⇒  P(K1) ≤ P(K2)    (puts non-decreasing in strike)
```

Strict rise `C(K2) > C(K1)` is immediate arb: buy low-strike call, sell high-strike call for positive cash and non-negative payoff.

### Intuition

A higher strike call gives you less — it should never cost more.

### Code Mapping

| Concept | Location |
|---|---|
| `ViolationType.MONOTONICITY` | `src/arbfree_vol/arbitrage/report.py:7` |
| Synthetic call construction | `src/arbfree_vol/arbitrage/quote_detect.py:131` `_normalize_to_calls()` / `:158` `_synthetic_call_price()` / `:183` `_parity_implied_call()` |
| Monotonicity check | `src/arbfree_vol/arbitrage/quote_detect.py:201` `_check_monotonicity()` |

### Detection Method

Check runs on **synthetic call prices** so put-only strikes participate:

1. Group by `K`. If call exists, use it; if put also exists, average call with parity-implied call `P + e^{-rT}(F-K)`. If no call, convert put via parity.
2. Sort by `K`, scan adjacent pairs: `jump = c2 - c1`. Flag if `jump > 1e-4` — offending quote is the higher-strike call.

Forward price (when via `detect_with_forward`) is threaded into parity conversion, so `r/q` errors do not alias as monotonicity violations.

### Tolerance

`1e-4` absolute (`quote_detect.py:213`). Pure price-level check, no `K`-scaling.

---

## 3. Butterfly / Convexity — `g(k) ≥ 0`

### Math

**Price convexity** (fixed `T`, `K1 < K2 < K3`):

```
C(K2) ≤ w·C(K1) + (1-w)·C(K3)   where w = (K3-K2)/(K3-K1)
```

Breeden–Litzenberger (1978) links this to the risk-neutral density (see §6):

```
∂²C/∂K² ≥ 0   ⇔   f_Q(K) ≥ 0
```

For SVI, Gatheral's `g(k)` is the parameterized form of `∂²C/∂K² ≥ 0`:

```python
# src/arbfree_vol/svi/model.py:18,29
core = svi_core(k, a,b,rho,m,sigma)  # returns (w, w', w'')
if core.w0 <= 0: return float("-inf")
g = (1 - k·w'/(2w))² - (w'²/4)(1/w + 1/4) + w''/2
```

No-butterfly ⇔ `g(k) ≥ 0  ∀ k`.

**Gatheral–Jacquier (2014) Theorem 4.2** — sufficient per-slice bounds for SSVI `(θ, ρ, ψ)`:

```
θ·ψ·(1+|ρ|)  <  4    (STRICT)       — condition 1
θ·ψ²·(1+|ρ|) ≤  4    (non-strict)   — condition 2
residual = min(4 - θψ(1+|ρ|), 4 - θψ²(1+|ρ|)) ≥ 0  ⇒  arb-free
```

Both are necessary for the proof; condition 1 is strict (`<`, not `≤`).

### Intuition

A butterfly spread (`long K1 + long K3 - 2·long K2`) pays non-negatively. Negative price = negative implied probability mass.

### Code Mapping

| Concept | Location |
|---|---|
| Raw SVI `w(k)` | `src/arbfree_vol/svi/model.py:40` `svi_total_variance()` — `w(k)=a+b(ρ(k-m)+√((k-m)²+σ²))` |
| `svi_core` `(w,w',w'')` | `src/arbfree_vol/svi/model.py:18` |
| `svi_g(k)` | `src/arbfree_vol/svi/model.py:29` |
| SSVI `w(k)` | `src/arbfree_vol/ssvi/model.py:60` `ssvi_w()` — `w=(θ/2)(1+ρψk+√((ψk+ρ)²+1-ρ²))` |
| eSSVI `ψ = η/θ^γ` | `src/arbfree_vol/ssvi/model.py:53` `essvi_psi()` |
| GJ condition | `src/arbfree_vol/ssvi/model.py:88` `gatheral_jacquier_condition(theta,rho,psi, strict=...)` |
| GJ strict epsilon | `src/arbfree_vol/ssvi/model.py:28` `_GJ_STRICT_EPS = 1e-9` (single source of truth) |
| Boundary-only helper | `src/arbfree_vol/ssvi/model.py:148` `essvi_params_in_bounds` (aliased `essvi_arb_safe`) — checks only `0≤γ≤1, η>0`, **not arb-free** (`issues.md` #9) |
| Quote-level butterfly | `src/arbfree_vol/arbitrage/quote_detect.py:226` `_check_butterfly()` |
| SVI curve butterfly | `src/arbfree_vol/arbitrage/svi_detect.py:37` `_check_butterfly()` |
| SVI `w_min` | `src/arbfree_vol/arbitrage/svi_detect.py:8` `min_total_variance()` — `w_min=a+bσ√(1-ρ²)` |
| SVI min-variance check | `src/arbfree_vol/arbitrage/svi_detect.py:18` `_check_min_variance()` |
| Optimizer GJ constraints | `src/arbfree_vol/ssvi/_butterfly.py:_butterfly_constraints` (re-exported as `ssvi/term_structure.py:68` `_butterfly_constraints`) |

### Detection Method

* **Quote-level** (`quote_detect.py:226`): discrete linear-interpolation test on synthetic calls — `line = w·c1+(1-w)·c3`, flag if `c2 - line > 1e-4` at middle `K2`.
* **Curve-level** (`svi_detect.py:37`): 121-point coarse scan over `[k_min,k_max]` default `[-3, 3]` to find approximate `argmin g(k)`, then `scipy.optimize.minimize_scalar(method='bounded')` in ±3-grid-step bracket (fix for `issues.md` #1 — bare Brent was trapped by the sharp `g(k)` spike at `k=m`).
* **`_check_min_variance`** (`svi_detect.py:18`): `w_min < -1e-4` ⇒ `NEGATIVE_VARIANCE`.
* **SSVI optimizer**: each `_hard_constraints` pair writes GJ as two smooth inequalities `(1+ρ)` and `(1-ρ)` separately. When `strict=True`, `gatheral_jacquier_condition` nudges condition 1's residual below zero if within `1e-9` of the boundary — exact equality is reported as violation. Same constant used in production constraints (`_GJ_CONDITION1_STRICT_EPS`), so diagnostic and calibration cannot diverge.

### Tolerance

`1e-4` for price convexity and `g(k)` / `w_min`. `1e-9` for GJ strict-boundary nudging.

---

## 4. Calendar Arbitrage — `w(k,T)` Non-Decreasing in `T`

### Math

For fixed log-moneyness `k = ln(K/F)`:

```
w(k, T) = σ_imp(k,T)² · T         (total variance)
T1 < T2  ⇒  w(k, T1) ≤ w(k, T2)   ∀ k
```

`w` is the natural arb-free state variable, not `σ` or `C`. Decrease ⇒ calendar spread arb.

**Hendriks & Martini (2019) Prop 3.1** — for eSSVI ordered by `T1<...<TN` with `χ_i = θ_i·ψ_i`:

```
(a) θ1 ≤ θ2 ≤ ... ≤ θN                                              (ATM variance)
(b) χ1 ≤ χ2 ≤ ... ≤ χN                                              (wing magnitude)
(c) |(ρ_{i+1}·χ_{i+1} - ρ_i·χ_i) / (χ_{i+1} - χ_i)| ≤ 1  ∀ i         (slope)
    plus both GJ butterfly bounds per slice
```

Per-slice `ρ` stays fully free (tanh-reparametrised, no cross-slice functional form) — `ssvi/term_structure.py:388` `fit_ssvi_surface_sequential`.

### Intuition

Total variance is cumulative uncertainty. Longer maturity cannot imply less cumulative variance at the same moneyness, or the term structure is internally inconsistent (short-dated event risk is the typical real-world culprit for apparent dips).

### Code Mapping

| Concept | Location |
|---|---|
| Total variance helper | `src/arbfree_vol/variance.py:slice_total_variance()` — `dict[strike, w]` with `w=σ²T` |
| Raw SVI soft calendar penalty | `src/arbfree_vol/svi/calibration.py:72` `_build_residuals()` — `√arb_penalty·√max(w_prev(k)-w(k),0)` when `prev_slice` threaded (see `repair/engine.py:_fit_slice` ascending-`T` order) |
| SVI curve calendar | `src/arbfree_vol/arbitrage/svi_detect.py:80` `_check_calendar()` |
| Quote-level calendar | `src/arbfree_vol/arbitrage/calendar.py:22` `_check_calendar()` + `quote_detect.py:4` import |
| eSSVI hard H&M constraints | `src/arbfree_vol/ssvi/term_structure.py:_hard_constraints` (Prop 3.1 a,b,c + GJ) |
| Sequential fitter | `src/arbfree_vol/ssvi/term_structure.py:388` `fit_ssvi_surface_sequential()` |
| H&M verifiers | `src/arbfree_vol/ssvi/_hm_verify.py` — `verify_hm_condition()`, `verify_ssvi_calendar_free()`, `verify_hm_condition_breakdown()` (re-exported via `term_structure.py:81`) |
| Constrained SVI entry | `src/arbfree_vol/svi/calibration.py:154` `calibrate_constrained(arb_penalty=100.0, prev_slice=None)` |
| Surface verifiers | `src/arbfree_vol/arbitrage/svi_detect.py:166` `detect_svi_surface()` / `:151` `detect_svi()` |

### Detection Method

* **Quote-level** (`calendar.py:22`): converts each slice to `(k, w)` via `slice_total_variance` + per-slice forward (`svi/data.py:_forward_price`), interpolates both slices onto common `k`-grid (overlap of their `k` ranges, `n_k=61` via `np.interp`), flags contiguous bands where `gap = w_earlier - w_later > 1e-4`.
* **Curve-level** (`svi_detect.py:80`): evaluates `w_earlier(k)` vs `w_later(k)` from `svi_total_variance` on default grid `linspace(-1.5, 1.5, 121)` (note: narrower than butterfly's `[-3,3]` — caveat in `issues.md` #4). Reports **contiguous violation bands** via `in_run/run_start/max_gap` state machine, not isolated points: `k=[a,b], worst gap=...`.
* **eSSVI optimizer**: `scipy.optimize.minimize` with hard inequality constraints (a,b,c). `fit_ssvi_surface_sequential` sorts by `T`, inherits `prev` = last *hard-constrained* success (fallback slices are not used as `prev`). Post-fit `verify_hm_condition` warning if any fallback broke monotonicity.

### Tolerance

`1e-4` for gap checks. Raw SVI penalty weight `arb_penalty=100.0` (`calibrate_constrained` default) with `√arb_penalty·√max(gap,0)` residual augmentation. eSSVI hard constraints use small eps floors `_EPS_THETA`, `_EPS_CHI` (`ssvi/_hm_margin.py`).

---

## 5. Breeden–Litzenberger Density

### Math

```
∂²C/∂K² = e^{-rT} · f_Q(K)    ≥  0
```

`f_Q(K)` = risk-neutral density of `S_T`. Negative second derivative ⇔ negative density ⇔ butterfly arb.

### Intuition

Call price curvature is the market's implied probability weight at that strike. Negative curvature = negative probability.

### Code Mapping

| Concept | Location |
|---|---|
| Continuous form (`g(k)`) | `src/arbfree_vol/svi/model.py:29` `svi_g()` — SVI-parameterized `g(k) ≥ 0` |
| Discrete form (quote-level) | `src/arbfree_vol/arbitrage/quote_detect.py:226` `_check_butterfly()` — positive second finite difference |
| No separate module | By design: `svi_detect.py:_check_butterfly` and `quote_detect.py:_check_butterfly` are the two instantiations |

### Detection Method

Same as §3 — no independent check. Quote-level convexity is the discrete proxy; `svi_g ≥ 0` is the continuous model-parametrized version. Dupire local vol (`src/arbfree_vol/pricing/local_vol.py:dupire_at`) also derives from this relation in `w` space.

---

## 6. Putting It Together — Repair Engine

```
repair(surface, use_ssvi=False, use_sabr=False)   # repair/engine.py
  → detect_with_forward() → rejection set → cleaned surface
  → estimate_forward_curve() → populate_per_slice_r()
  → SVI:   calibrate_constrained(..., prev_slice=...) ascending T  — soft calendar penalty
    eSSVI: fit_ssvi_surface_sequential()           — hard H&M Prop 3.1 + GJ bounds
    SABR:  fit_sabr_term_structure()               — B-spline + soft calendar (empirical, not arb-free by construction)
  → detect_svi_surface()                           — load-bearing verification
  → RepairReport(rejected, fitted_slices, remaining_violations,
                 fallback_slices, failed_slices, repair_infeasible)   # repair/report.py:33
```

Selecting the result: `use_ssvi` and `use_sabr` are mutually exclusive. Downstream `iv_at` / Greeks / Dupire / plots consume `FittedSurface` via `ssvi/model.py:168` or `sabr/model.py:123` `to_raw_svi_params()` adapters unchanged.

---

## 7. Caveats — What Is Certified, What Is Not

### Grid-Based Certificate vs Global Analytic Guarantee

Hard-constrained eSSVI slices satisfy the **implemented** GJ conditions and the **grid-based** calendar check — a *discrete* certificate, not a closed-form analytic one. A pair can pass every parameter inequality yet cross between grid points or beyond the grid.

**Grids used for verification:**

| Layer | Butterfy grid | Calendar grid |
|---|---|---|
| `svi_detect.py` | `[-3.0, 3.0]`, 121 pts + local Brent (`svi_detect.py:37`) | `[-1.5, 1.5]`, 121 pts (`svi_detect.py:82`) — **narrower than butterfly** (`issues.md` #4) |
| Native SSVI post-fit gate (`verify_ssvi_calendar_free`) | — | `[-3, 3]`, 241 pts (`ssvi/_hm_verify.py`) |
| Quote-level calendar | — | Overlap of slices' observed `k` ranges, 61 pts (`calendar.py:22`) |

Consequence: wing-only violations with `|k| > 1.5` are missed by `detect_svi_surface` at defaults; pass a wider `k_grid` explicitly when auditing wings.

### The OPEN Counterexample (`issues.md` § "eSSVI calendar certificate is grid-based, not a global analytic guarantee")

From `docs/architecture_review.md` Finding 3 (mitigated 2026-08-05, still OPEN at HEAD — FIX-6 probe 2026-08-12):

```
p1 = SSVIParams(theta=0.0149505446, rho=-0.6548551, psi=0.11491999)
p2 = SSVIParams(theta=0.0574982989, rho=-0.8830506, psi=2.5226500)
p1 GJ residuals: [3.999407, 3.99715677, 3.99993185, 3.99967326]  (all ≥0)
p2 GJ residuals: [3.98303671, 3.72686712, 3.95720757, 3.31098134] (all ≥0)
χ = θψ = 0.0017181164, 0.1450480837  (non-decreasing)
verify_hm_condition: True            (a,b,c all pass)
w2 - w1 at k=0.68 = -0.0015283049    ← calendar-spread violation
HM sufficient disjunction (Prop 3.5 restatement): phi_ratio 21.95 > 1, lhs 5460.42 > rhs 5271.15 → False
```

At HEAD this pair is **not silently certified**: `verify_ssvi_calendar_free` over `linspace(-3,3,241)` returns `False`, so `repair()` sets `repair_infeasible=True` and surfaces the violation. The mitigation is **grid-based** — it depends on the violation landing on the grid (here it does: `min w2-w1 = -0.0015288` at `k≈0.675`). Violations strictly between points or beyond `[-3,3]` are not certified. Resolving the exact H&M Prop 3.1 / Prop 3.5 statement and adding the sufficient disjunction to `verify_hm_condition` remains **OPEN**.

### Fallback Slices — Not Certified

On live SPY data 4–7 of ~20 slices fall back to the unconstrained per-slice fit because ATM total variance `θ = σ²T` dips non-monotonically in the short end (event risk, microstructure). Example dip at `T=0.0658→0.0849`: `θ` drops 29% (`issues.md` #15). The H&M condition (a) requires `θ` non-decreasing; when data wants a dip, the constrained optimizer reports "Positive directional derivative for linesearch" (saddle/boundary). Warm-starting from the unconstrained solution fails 100% (it violates (a) itself); random restarts converge to degenerate copies of `prev` with 1.1–3.6× worse RMSE.

```python
# src/arbfree_vol/repair/report.py:33,42-43
@dataclass
class RepairReport:
    repair_infeasible: bool = False      # True if any eSSVI hard fit failed
    fallback_slices: list[float]         # Ts that fell back — NOT arb-free
    failed_slices: list[float]           # Ts where both fits failed
```

`fit_ssvi_surface_sequential` routes fallback slices honestly: they remain in `fitted_slices` but are listed in `fallback_slices`, `repair_infeasible=True`, and `detect_svi_surface` reports remaining `CALENDAR` violations. The post-fit `verify_hm_condition` warning and Dupire NaN-propagation (`pricing/local_vol.py` stencil check via `fallback_slices`) rely on this flag. SABR (`use_sabr=True`) is *always* empirical — soft penalty + grid check, no closed-form guarantee; dynamic SABR not implemented (`issues.md` #14).

**Interview line:** "Hard-constrained eSSVI is arb-free on the grid we check; fallback and SABR slices are not. We report that honestly."

---

## 8. Interview Quick Answers

| Question | Answer |
|---|---|
| Why `w = σ²T` not `σ`? | `w` is the arb-free state variable; `σ` mixes time scaling. Calendar arb is `w(k,T)` ↘, not directly visible in `σ` or `C`. |
| Why per-slice `r/q`? | `r/q` vary by `T`; surface-level `r/q` creates systematic parity bias (~$0.67 on SPY). Estimate forwards via median parity, back out per-slice `r`. |
| Why median not mean for forwards? | One bad quote corrupts the mean; median is robust. |
| `g(k)` spike trap? | `g(k)` spikes to +300 at `k=m` for tight `σ`; bare Brent brackets the wrong side. Fix: 121-pt coarse scan + local Brent in ±3 steps. |
| Calendar narrow grid? | `svi_detect` calendar `[-1.5,1.5]` misses wing violations; pass wider `k_grid` or rely on `verify_ssvi_calendar_free` `[-3,3]`. |
| What does GJ guarantee? | Sufficient (not necessary) per-slice no-butterfly when both `θψ(1+|ρ|)<4` and `θψ²(1+|ρ|)≤4`. |
| H&M Prop 3.1 sufficient+necessary? | H&M (a)(b)(c) is **sufficient** for no calendar spread; a dip in `θ` can still be arb-free with skew adjustment — hence fallback pathology. |
| What is `repair_infeasible`? | `True` when any eSSVI hard fit failed or a grid gate tripped — the surface is not fully certified even if `remaining_violations` is empty on a narrower grid. |

---

## References

* Breeden, D. & Litzenberger, R. (1978). Prices of State-Contingent Claims Implicit in Option Prices. *J. Business* 51(4).
* Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives.
* Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance* 14(1), 59–71.
* Hendriks, S. & Martini, C. (2019). The Extended SSVI Volatility Surface. *J. Computational Finance* 22(5), Prop 3.1.
* Corbetta, J. et al. (2019). Robust calibration and arbitrage-free interpolation of SSVI slices. *arXiv:1804.04924*, Sec 2.2–2.3.
* Hagan, P. et al. (2002). Managing smile risk. *Wilmott*, SABR Hagan eq. A.69a/b (implemented in `sabr/model.py:57`).
