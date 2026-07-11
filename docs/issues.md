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
