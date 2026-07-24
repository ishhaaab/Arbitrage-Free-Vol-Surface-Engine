# Known Issues

## 1. `_check_butterfly` — `minimize_scalar(bounded)` can miss violations near SVI spike

**File:** `src/arbfree_vol/arbitrage/svi_detect.py:_check_butterfly`

**Problem:**
`minimize_scalar(method='bounded')` is a local optimizer. When `g(k)` has a sharp spike (e.g. at `k = m` where `g` shoots to +300+), Brent's bracket gets trapped and returns a boundary point, completely missing violations on the other side of the spike.

**Example params that trigger this:**
```python
SVIParams(a=0.04, b=0.9, rho=0.97, m=2.0, sigma=0.01)
```
- `g(k)` is positive everywhere in `[-2, 2]`
- `g(k)` has a huge spike (+323) at `k = 2.0`
- `g(k)` goes negative at `k ≈ 2.7` (inside `[-3, 3]`)
- `minimize_scalar(bounded)` returns `k = -3.0, g ≈ +0.59` — misses the violation

**Fix (when ready):**
Replace the bare `minimize_scalar` call with a **coarse grid scan + local refinement** approach:
1. Evaluate `g(k)` on a coarse grid (e.g. 121 points over `[k_min, k_max]`).
2. Find the grid point with minimum `g`.
3. Refine locally with `minimize_scalar(bounded)` in a narrow bracket (±3 grid spacings) around that point.

121 extra `svi_g` evaluations is negligible cost. The function signature and output contract do not change.

**Status:** Fixed. Grid scan (121 points) + local bounded refinement applied. All 13 tests pass.

---

## 2. Parity check uses surface-level r/q — systematic false positives on real data

**File:** `src/arbfree_vol/arbitrage/quote_detect.py:_check_parity`

**Problem:**
The parity residual `|C - P - (S·e^{−qT} − K·e^{−rT})|` uses surface-level `r` and `q` from the `VolSurface`. On real market data (SPY), the true risk-free rate and dividend yield differ from the defaults (`r=0.05`, `q=0.0`), creating a systematic bias of ~$0.50–$0.80 per strike.

Example from SPY (spot=754, T=0.0274, r=0.05, q=0.0):

| Strike | C | P | C-P | Forward | Residual |
|--------|---|---|-----|---------|----------|
| 755 | 4.45 | 4.76 | -0.31 | 0.36 | -0.67 |

The residual ($0.67) is far larger than the bid-ask spread ($0.02), so every strike gets flagged — **~95% rejection rate** on a surface that should mostly be valid.

**Root cause:**
Surface-level `r` and `q` are approximations. Real interest rates and dividends vary by expiry and asset. Without correct rates, the parity check generates false positives across the whole surface.

**Fix (planned — see docs/plan-parity-rq.md or issue conversation):**
1. Try to fetch real `r` and `q` from yfinance (`^IRX` for risk-free rate, `info.dividendYield` for dividend yield).
2. When real rates are unavailable or the carry is unknown, run `estimate_forward_curve()` as a pre-pass before detection and thread the per-expiry forward price into `_check_parity`.
3. The threshold stays market-aware (`max(half_spread_C, half_spread_P, 0.05)`) — the fix is about the *reference price* (the forward), not the tolerance.

**Status:** Fixed. Implemented three-layer approach:
1. Real r/q from yfinance (`^IRX` + `info.dividendYield`) with sanity cap.
2. `detect_with_forward()` pre-pass: estimates per-expiry forward via median of strike-level parity, threads into `_check_parity`.
3. Median-based aggregation in `_slice_forward` (not mean) for robustness to outliers.
Total parity rejection on SPY dropped from ~95% to ~45% (remaining violations are genuine strike-level inconsistencies, not systematic bias).

---

## 3. SVI butterfly breakdown on near-expiry data with steep right wing

**Files:**
- `src/arbfree_vol/svi/model.py:svi_g` — the density condition `g(k) ≥ 0`
- `src/arbfree_vol/arbitrage/svi_detect.py:_check_butterfly` — min-finding via grid+refine

**Problem:**
On very short-dated options (T < 0.05 years ≈ 18 days), the implied volatility smile becomes steep, especially in the right wing (OTM calls on SPY). SVI can fit the smile but produces a negative risk-neutral density (`g(k) < 0`) beyond k ≈ 0.5–1.5, even after cleaning and repair.

Example from SPY demo:
```
SVI butterfly arbitrage: g=-14.88 < 0 at k=0.2473 (negative risk-neutral density)
SVI butterfly arbitrage: g=-90.45 < 0 at k=1.8061 (negative risk-neutral density)
```

These are **model violations, not quote violations** — the SVI parameterization is being pushed to fit a steep smile that violates convexity at the wings.

**Root cause:**
Raw SVI is expressive but not arbitrage-free by construction. For near-expiry data with high skew (equity puts are expensive, calls are cheap), the calibrated `rho` (skew) and `b` (spread) can produce negative density at the wings even though the fit is visually good. This is a known limitation of raw SVI — see Gatheral (2004), Gatheral & Jacquier (2014).

**Possible mitigations (none yet implemented):**
1. **Fit with a butterfly penalty** — add `g(k) < 0` as a soft constraint in the `least_squares` objective.
2. **Parameter constraints** — restrict `b * (1 + |rho|) < 2 * a / sigma` (Gatheral & Jacquier condition for SSVI).
3. **Upgrade to SSVI/eSSVI** — the surface parameterization that is arbitrage-free by construction (deferred, listed in Project.md).
4. **Increase minimum time-to-expiry** for data fed into the SVI calibrator (pragmatic — skip T < 14d).

**Status:** Known, documented. Not yet mitigated. The repair engine reports these honestly as remaining violations.

---

## 4. SVI calendar check narrower k-range than butterfly check

**File:** `src/arbfree_vol/arbitrage/svi_detect.py:_check_calendar`

**Problem:**
The default `k_grid` scans `[-1.5, 1.5]` while `_check_butterfly` scans `[-3.0, 3.0]`. Calendar arbitrage violations that manifest only in the deep wings (`|k| > 1.5`) will be missed at default settings. Callers can widen the range via the `k_grid` parameter, but the default is inconsistent with the butterfly check.

**Status:** Known, not yet mitigated.

---

## 5. Hardcoded dummy expiry date in `slice_total_variance`

**File:** `src/arbfree_vol/variance.py:30`

**Problem:**
Every `OptionContract` is constructed with `expiry_date=date(2004, 1, 1)`. Currently harmless because `expiry_time` (float) is the value used in all pricing — the date field is unused. If any future refactor computes T from the calendar date, every contract will silently get the wrong time-to-expiry.

**Suggested fix:** Use `date.min` instead of an arbitrary date so any accidental dependency on the date field fails loudly (a zero or negative T) rather than producing subtly wrong results.

**Status:** Known, not yet fixed.

---

## 6. Negative total variance silently masked as 0.0 in visualization

**File:** `src/arbfree_vol/viz/surface.py:42,51`

**Problem:**
`sqrt(w / T) if w > 0 else 0.0` under `np.errstate(divide="ignore", invalid="ignore")`. If SVI calibration produces negative total variance (an arbitrage violation caught by detection), the plots show flat `0.0` volatility, hiding the math failure from the user. The detection pipeline correctly reports the violation, but the visualization lies.

**Status:** Known, not yet fixed (visualization-only impact).

---

## 7. Hardcoded strike boundaries in `plot_iv_heatmap`

**File:** `src/arbfree_vol/viz/surface.py:176`

**Problem:**
`strikes = np.linspace(fs.spot * 0.8, fs.spot * 1.2, n_strikes)` hardcodes the strike range to `[80%, 120%]` of spot. For volatile assets or long-dated maturities where the surface extends significantly beyond ±20%, the heatmap silently clips the wings. The underlying `iv_at()` works correctly at any valid strike — this is purely a visualization limitation.

**Status:** Known, not yet fixed (visualization-only impact).

---

## 8. Missing financing costs in delta-hedged backtest P&L

**File:** `src/arbfree_vol/backtest/pnl.py:110-133`

**Problem:**
The daily hedge P&L loop computes only `(-qty * delta) * (S_curr - S_prev)` — the stock price change. Missing from the realized P&L are:
- Interest paid/received on the option premium (`qty * option_price * r * dt`)
- Interest paid/received on the hedge position (`-qty * delta * S * r * dt`)
- Dividend adjustment on the short stock hedge (`-(-qty * delta * S) * q * dt`)

The docstring states "frozen-vol convention" and "standard simplification," but those refer to the *hedge vol*, not financing costs. Financing is a fundamental component of P&L. For a 30-day SPY trade (spot=$500, sigma=0.20, r=5%, q=1.5%), the missing financing per trade is on the order of $0.50-$1.00 — material relative to typical mispricing trade P&L.

**Suggested fix:** Add daily financing cashflows to the hedge loop:
```
hedge_pnl += (-qty * delta_prev) * (S_curr - S_prev)
hedge_pnl += (-qty * option_value) * r * dt_actual
hedge_pnl += (qty * delta_prev * S_prev) * r * dt_actual
hedge_pnl += (-qty * delta_prev * S_prev) * q * dt_actual
```
Or explicitly document that this implementation intentionally excludes carry for simplicity and caveat the reported P&L.

**Status:** Known, not yet fixed.

---

## 9. `essvi_arb_safe` is a structural bounds check, not an arb-free guarantee

**File:** `src/arbfree_vol/ssvi/model.py:94`

**Problem:**
Returns `True` when `0 <= gamma <= 1` and `eta > 0` — necessary but not sufficient for arbitrage-free eSSVI surfaces. The full Gatheral-Jacquier condition `theta * psi(theta) * (1 + |rho|) <= 4` must be evaluated per-slice across the surface. The docstring was corrected in an earlier fix to be honest about this, but the function name `essvi_arb_safe` remains misleading.

**Suggested fix:** Rename to `essvi_params_in_bounds` so the name matches what the function actually checks.

**Status:** Known, not yet fixed.

---

## 10. Expired option clamping in CSV loader

**File:** `src/arbfree_vol/ingestion/loader.py:32`

**Problem:**
`max(0.0, days / 365.0)` clamps negative days (expired options) to `T = 0.0`. If an expired option bypasses near-expiry filtering, this produces `ExpirySlice(expiry_time=0.0)` which fails Pydantic validation (`Field(gt=0)`) — a confusing error message rather than a clear "option is expired" rejection. The near-expiry cleaning rule (`min_T = 7/365`) guards against this in normal paths, but the clamping is fragile and should either reject with a clear cause or raise before creating the slice.

**Status:** Known, not yet fixed (defended by existing near-expiry filtering).

---

## 11. `total_variance_at` interpolates at fixed strike K — not fixed log-moneyness

**File:** `src/arbfree_vol/surface/interpolate.py:177-188`

**Problem:**
Linear interpolation in T at a fixed absolute strike K means the two endpoints are evaluated at different log-moneyness values (`k_low = log(K/F_low) ≠ log(K/F_high) = k_high` since forward prices differ across expiries). For calendar arbitrage (`∂w/∂T ≥ 0`), the condition should be checked at consistent k, not consistent K. For typical equity parameters (`r - q ≈ 3%`, `dT ≈ 1e-3`), the k-drift is ~3e-5 and the resulting w-error from smile skew is O(1e-6) — well below the `1e-4` arb tolerance. This is a theoretical limitation, not an active bug with standard parameters.

**Status:** Known, not yet fixed (negligible numerical impact with typical parameters).

---

## 12. SABR→SVI mapping RMSE not exposed

**File:** `src/arbfree_vol/sabr/model.py:to_raw_svi_params`

**Problem:**
`to_raw_svi_params` fits raw SVI parameters to a SABR smile via `least_squares` but returns only the fitted `(a, b, rho, m, sigma)` tuple — not the RMSE of the approximation. The docstring states "the caller should inspect rmse if accuracy is critical," but the API provides no way to access it. In the repair pipeline, the RMSE stored in `FittedSlice` is the SABR-to-data error, not the SVI-to-SABR approximation error. Users of the mapped parameters cannot judge how well SVI represents the SABR smile.

**Status:** Known, not yet fixed (feature request).

---

## 13. Backtest trades use surface-level r/q, ignoring per-slice overrides

**File:** `src/arbfree_vol/backtest/engine.py:89-90`

**Problem:**
`run_backtest` constructs `Trade` objects with `risk_free=surface.risk_free` and `div_yield=surface.div_yield`. However `detect_mispricing` uses per-slice rates via `get_r(surface, sl)` / `get_q(surface, sl)` — which may differ from surface-level defaults after `populate_per_slice_r`. The signal detection and trade realization thus use different discount/forward rates, producing internally inconsistent P&L.

**Status:** Known, not yet fixed. Mitigation: surface-level r/q are reasonable approximations for liquid equities; the `detect_with_forward` pre-pass corrects the worst cases. Fix would require threading per-slice rates through `MispricingSignal` and `Trade`.
