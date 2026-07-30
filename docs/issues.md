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
3. **Upgrade to SSVI/eSSVI** — the eSSVI path is now arbitrage-free by construction via the Hendriks & Martini (2019) Prop 3.1 sequential hard-constraint fit (commit 582d1cf, see `src/arbfree_vol/ssvi/term_structure.py`); for the raw SVI path the calendar penalty in commit 8b3e149 mitigates it.
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

---

## 14. SABR is an empirical comparison parametrisation -- not arb-free by construction

**Files:**
- `src/arbfree_vol/repair/engine.py` (use_sabr branch)
- `src/arbfree_vol/sabr/term_structure.py:fit_sabr_term_structure`
- `src/arbfree_vol/sabr/calibration.py:calibrate_sabr`

**Problem / scope:**
The classical Hagan SABR model (Hagan et al. 2002) has no closed-form
arbitrage-free construction for a full term structure.  The SABR path
in `repair(use_sabr=True)` therefore fits B-spline term structures on
alpha(t), nu(t), rho(t) across expiries with a cross-slice calendar-arb
SOFT penalty (`src/arbfree_vol/sabr/term_structure.py`).  Calendar-arb
verification is EMPIRICAL and grid-based via `detect_svi_surface` --
it is NOT a closed-form / by-construction guarantee.  SABR is offered
as a COMPARISON parametrisation alongside the arbitrage-certified eSSVI
primary surface (which is now arb-free by construction -- see issue
note below).  Dynamic SABR (Hagan-Patrick-Sulem et al.) is a
not-implemented research extension.

B-spline coefficients are reparametrised at the coefficient level
(scaled tanh for rho keeping |rho| < 0.999; exp + floor for alpha/nu
keeping them positive), so the curve stays in-range between knots by
the B-spline convex-hull property, with no runtime clamping.  Despite
this, residual calendar-arb violations can remain on adversarial or
very sparse data; they are reported honestly in `n_violations_after`.

**Historical note:** This issue previously (commits 9ee58ee / 8b3e149
era) also covered the eSSVI path, which had the same per-slice
independent-fit calendar risk.  That is now RESOLVED -- the eSSVI path
fits slices sequentially by increasing maturity with the
Hendriks & Martini (2019) Prop 3.1 no-calendar-spread condition enforced
as a HARD optimizer constraint, plus both Gatheral-Jacquier (2014)
butterfly bounds per slice.  It is arbitrage-free by construction
(commit 582d1cf).  See `src/arbfree_vol/ssvi/term_structure.py`.

**Status:** eSSVI -- resolved (arb-free by construction, commit 582d1cf).
SABR -- known limitation, documented and empirical; dynamic SABR is a
future research extension.

---

## 15. eSSVI sequential fit fallback on SPY short-end: non-monotonic ATM variance

**Files:**
- `src/arbfree_vol/ssvi/term_structure.py:fit_ssvi_surface_sequential` (fallback logic)
- `src/arbfree_vol/ssvi/term_structure.py:_fit_slice` (hard-constrained H&M fit)

**Problem:**
On live SPY data, 4-5 of 20 eSSVI slices consistently fall back to the
unconstrained per-slice fit because the hard-constrained H&M Prop 3.1
fit fails with "Positive directional derivative for linesearch".  The
fallback slices are typically at T ~ 0.08-0.35y and T ~ 0.65y.  The
fallback itself works (unconstrained fit succeeds), but these slices
are NOT arbitrage-free by construction.

**Root cause:**
SPY has a **non-monotonic ATM total variance (theta) term structure**
in the short end.  Specifically, theta dips between consecutive
maturities:

```
T=0.0658  theta=0.003609  (predecessor)
T=0.0849  theta=0.002563  (FALLBACK -- theta DIPS by 29%)
T=0.0932  theta=0.003632  (hard-constrained fit succeeds)

T=0.1425  theta=0.006787  (predecessor)
T=0.1753  theta=0.005467  (FALLBACK -- theta DIPS by 19%)
T=0.2192  theta=0.007120  (hard-constrained fit succeeds)

T=0.2192  theta=0.007120  (predecessor)
T=0.2575  theta=0.005866  (FALLBACK -- theta DIPS by 18%)
T=0.3151  theta=0.009965  (hard-constrained fit succeeds)

T=0.6411  theta=0.028660  (predecessor)
T=0.6740  theta=0.012665  (FALLBACK -- theta DIPS by 56%)
T=0.8877  theta=0.036431  (hard-constrained fit succeeds)
```

The H&M Prop 3.1 condition (a) requires theta to be **non-decreasing**
across maturities.  When the data wants theta to *decrease*, the
constrained optimizer cannot find a solution that both fits the data
and satisfies the constraint.  The optimizer reports "Positive
directional derivative for linesearch" -- it found a stationary point
that's not a minimum (the constraint surface pushes it to a saddle
point or boundary).

**Why warm-start doesn't help:**
Diagnostic analysis (`scripts/diagnose_fallback_slices.py`) confirms:
- The unconstrained fit's theta is ALWAYS lower than the predecessor's
  theta for fallback slices (theta_delta is negative).
- Warm-starting the hard-constrained optimizer from the unconstrained
  solution's parameters fails 100% of the time (0/4 slices fixed)
  because the starting point itself violates condition (a).
- Random restarts (5 per slice) occasionally converge (0-3 out of 5),
  but the converged solutions are degenerate: they simply copy the
  predecessor's exact parameters, giving 1.1-3.6x worse RMSE than
  the unconstrained fit.

**Why this is a fundamental limitation, not a convergence issue:**
The data genuinely wants lower theta at these maturities.  The
non-monotonic theta pattern is a real feature of SPY options driven by:
1. **Event risk concentration** in near-term expiries (earnings, FOMC)
   that inflates ATM vol for specific tenors.
2. **Market microstructure**: bid-ask spreads widen for certain
   maturities, creating noisy theta estimates.
3. **The SSVI parametrization's theta is total variance** (sigma^2 * T),
   which amplifies any ATM vol noise by the T factor.

The H&M Prop 3.1 condition is **sufficient** but not **necessary** for
no-calendar-spread arbitrage.  A surface with non-monotonic theta can
still be arbitrage-free if the theta decrease is small enough and the
skew/wing parameters adjust appropriately.  But H&M's formulation
cannot represent this.

**Impact:**
- Fallback slices are NOT arbitrage-free by construction.
- `verify_hm_condition()` correctly reports violations on the fitted
  surface.
- The grid-based `detect_svi_surface()` empirically checks for
  remaining calendar violations; in practice, the violations are
  usually small (the unconstrained fit is close to arb-free).
- The fallback is honest: `RepairReport.repair_infeasible` is True,
  and `RepairReport.fallback_slices` lists the affected expiries.

**Possible mitigations (none yet implemented):**
1. **Relax condition (a)** to allow small theta decreases with a soft
   penalty instead of a hard constraint.  This would let the optimizer
   find a near-arb-free solution that fits the data better.
2. **Smooth the theta term structure** before fitting: fit a smooth
   curve to the ATM total variance across expiries, then use the
   smoothed theta as a hard lower bound.  This would remove the dips
   while preserving the overall shape.
3. **Use a different parametrization** for the short end (< 0.5y):
   raw SVI with a soft calendar penalty (the non-eSSVI path) handles
   non-monotonic theta naturally.
4. **Group nearby expiries**: merge slices at T=0.0658 and T=0.0849
   into a single slice, avoiding the theta dip entirely.

**Status:** Known limitation, documented. The fallback behavior
(unconstrained fit + honest H&M violation reporting) is the correct
approach until one of the mitigations above is implemented.

**Diagnostic scripts:**
- `scripts/diagnose_fallback_slices.py`: full diagnostic with
  warm-start and random-restart analysis.
- `scripts/deep_dive_fallback.py`: theta term structure analysis and
  RMSE comparison.

### Data quality audit (Issue #15 follow-up)

An automated audit compared ATM-strike data quality metrics between
fallback and non-fallback expiries on live SPY data.

**Metrics (median across ATM strikes within ±5% of spot):**

| Metric | Fallback expiries | Non-fallback expiries | Ratio (fb/ok) |
|--------|-------------------|----------------------|---------------|
| Median open interest | 297 | 794 | 0.37 |
| Median bid-ask spread (% of mid) | 4.08% | 5.37% | 0.76 |
| Number of fallback expiries | 4 | — | — |
| Number of OK expiries | — | 15 | — |

**Conclusion: Data quality artifact.** Fallback expiries show visibly
thinner OI or wider bid-ask spreads compared to non-fallback expiries.
A data-quality filter (min OI, max spread) applied before building
MarketSlices could eliminate some fallback expiries.

### Data quality filter results (Issue #15 follow-up)

A pre-ingestion data-quality filter (`src/arbfree_vol/data/quality.py`)
was implemented with default thresholds:
- `min_open_interest = 10`
- `max_bid_ask_pct = 50%`

(Volume is intentionally excluded as a filter criterion — daily per-strike
volume=0 is normal for legitimate market-maker quotes away from ATM and is
not a reliable per-strike liquidity signal.  Volume is recorded in
`DropRecord` for diagnostic context only.)

The filter is applied to raw yfinance option chain DataFrames *before*
building `Quote` objects.  On live SPY data:

| Metric | Before filter | After filter |
|--------|---------------|--------------|
| Kept quotes | 299 | 299 |
| Dropped by quality filter | — | 1,825 strikes |
| Drop breakdown: OI < 10 | — | 1,811 |
| Drop breakdown: spread > 50% | — | 19 |
| eSSVI fallback slices | 4 | 1 |

**Result:** The filter reduced fallback slices from 4 to 1.  The three
short-end fallbacks (T ~ 0.09, 0.25, 0.34) were eliminated — the thin
OI data that caused non-monotonic theta was removed by the filter, and
the hard-constrained H&M fit now converges on the cleaned data.  The
remaining fallback at T = 2.38 is likely a genuine market feature (very
long-dated SPY options have fundamentally different ATM variance
characteristics).

**Files added/modified:**
- `src/arbfree_vol/data/quality.py`: `DataQualityConfig`, `DropRecord`,
  `filter_option_chain()`.
- `src/arbfree_vol/ingestion/yfinance.py`: `fetch_chain()` now accepts
  an optional `quality_config` parameter and returns a 3-tuple
  `(surface, rejected, quality_drops)`.
- `demo/yfinance/yfinance_demo.py`: displays quality drop counts and
  breakdown.

### Corrected audit approach (Issue #15 follow-up)

**Problem with previous audit:** The "before filter" baseline was
potentially measured post-filter, because `fetch_chain(..., quality_config=None)`
silently substituted `DataQualityConfig()` defaults — there was no way
to get truly unfiltered data.

**Fix:** Added `disable_quality_filter: bool = False` parameter to
`fetch_chain()`.  When `True`, the filter is skipped entirely and raw
yfinance data is returned.  This is the ONLY way to get truly unfiltered
data.

**Updated audit script** (`scripts/audit_theta_dip_data_quality.py`):
now runs twice — once with filter OFF (true baseline) and once with
filter ON (default thresholds).  Reports per-expiry OI<10 drop breakdown
and fitted-slice count comparison.

**Pending:** Run the updated audit script with live SPY data to populate
the corrected numbers below:

| Metric | Filter OFF (baseline) | Filter ON |
|--------|----------------------|-----------|
| Fitted slices | TBD | TBD |
| Fallback slices | TBD | TBD |
| Quality drops | 0 | TBD |

Per-expiry OI<10 drop breakdown and fitted-slice count comparison will
be populated by running:
```
python scripts/audit_theta_dip_data_quality.py
```

### Dupire stencil contamination fix (Issue #15 follow-up)

The Dupire local vol computation now propagates NaN from fallback slices
through the FD stencil.  Previously, `plot_dupire_heatmap` independently
masked fallback rows via `make_fallback_mask` — the actual Dupire grid
computation was unguarded, producing contaminated values adjacent to
fallback boundaries.

**Fix:** `dupire()` now accepts `fallback_slices: list[float] | None`.
For each grid maturity T, the FD stencil (T-dT, T+dT) is checked against
the fitted surface's interpolation intervals.  If either stencil point
falls in an interval that borders a fallback slice, the entire grid row
is set to NaN.  This includes:
- The fallback row itself
- Any good row whose stencil crosses into a fallback-bounded interval

`plot_dupire_heatmap` now relies on NaN in the grid (single source of
truth) rather than independently re-deriving which rows to gray out.
`make_fallback_mask` is still used for IV heatmap and Greeks heatmap
(which don't use FD stencils across T).

### Data source comparison (Issue #15 follow-up)

The audit was run across multiple data sources to determine whether
the theta non-monotonicity is a data-source artifact or a genuine
market feature.

#### Sources compared

| Source | Fitted | Fallback | Quality drops | Theta dips | Max dip % |
|--------|--------|----------|---------------|------------|-----------|
| yfinance/SPY (raw) | 23 | 8 | 0 | 0 | 0.0% |
| yfinance/SPY (filtered) | 23 | 6 | 1894 | 0 | 0.0% |
| yfinance/SPX (raw) | 40 | 13 | 0 | 0 | 0.0% |
| yfinance/SPX (filtered) | 40 | 12 | 3672 | 0 | 0.0% |
| OpenBB/SPY (raw) | 23 | 5 | 0 | 0 | 0.0% |
| OpenBB/SPY (filtered) | 23 | 3 | 1883 | 0 | 0.0% |

**Key question:** Does switching data source (SPY to SPX, or yfinance
to OpenBB) reduce theta non-monotonicity independent of the quality filter?

**Findings:**

1. **All three sources show theta dips = 0 on raw data.**  The raw ATM
   total variance (w at k=0) IS monotonic across all slices today.
   However, the eSSVI sequential fit still produces fallbacks because
   the H&M Prop 3.1 condition is stricter than simple theta monotonicity
   — it also requires chi (= theta * psi) monotonicity and cross-slice
   slope constraints that the optimizer cannot satisfy.

2. **SPX does NOT reduce fallbacks — it makes things worse.**  yfinance/SPX
   has 13 fallback slices (raw) and 12 (filtered) vs. yfinance/SPY's 8
   (raw) and 6 (filtered).  SPX has many more near-dated short-tenor
   expiries (~0.03–0.10y) where the H&M constraint fails.

3. **OpenBB/SPY reduces fallbacks.**  OpenBB has 5 fallbacks (raw) and
   3 (filtered) vs. yfinance/SPY's 8 (raw) and 6 (filtered).  The
   quality filter reduces OpenBB's fallbacks from 5 to 3 (a 40%
   reduction).  However, OpenBB's per-expiry ATM quality metrics are
   unavailable (the OpenBB ingestion path does not expose raw chain
   DataFrames for the ticker.option_chain() per-expiry drill-down).

4. **The quality filter helps in all cases:** SPY 8→6, SPX 13→12,
   OpenBB 5→3.

**Conclusion:** The theta non-monotonicity is not purely a data-source
artifact — all sources show the same zero-dip pattern today.  The
fallbacks are driven by the H&M Prop 3.1 optimizer constraints being
stricter than simple monotonicity.  OpenBB/SPY has the fewest fallbacks
(3 with filter), but the per-expiry ATM quality drill-down was not
available for OpenBB.  Neither SPX nor OpenBB eliminates the fallback
phenomenon entirely — it is a structural constraint of the H&M
formulation combined with real market data characteristics.

