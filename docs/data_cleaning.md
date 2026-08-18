# Data Cleaning — Quote Quality & Rejection Rules

> Raw option markets are noisy. A volatility surface that ingests them naively will fit noise, not signal — and violate arbitrage in the process. This document explains the two-layer defense this project uses to turn retail option feeds into a fittable surface, and why every threshold is a named, configurable parameter instead of a magic number.

---

## 1. Why Option Data Is Noisy

An option chain is not a clean mathematical function `C(K,T)`. It is a market-microstructure artifact produced by fragmented venues, competing market makers, and uneven incentives to quote. Five sources dominate:

| Source | What happens | Impact on `w(k,T)` |
|---|---|---|
| **Market microstructure** | Quotes are two-sided (`bid`/`ask`) set by market makers who widen spreads where they carry inventory risk. The `mid` you invert for IV is an estimate, not a trade. | Wide-spread mids inject skew noise; inverting a stale mid produces a phantom IV. |
| **Uneven liquidity** | ATM strikes trade thousands of contracts/day; wings (deep OTM/ITM) may print zero trades all day but still show a posted quote. | Wing IVs are inferred from quotes no one would actually trade at. SVI will bend to fit them. |
| **Stale quotes** | A market maker posts a quote at 09:35 and leaves it; the underlying moves 1% but the quote does not. `lastPrice` may be hours old. | IV computed from a stale price implies a vol that never existed. Calendar and butterfly checks then fire on ghosts. |
| **Bad prints / crossed markets** | Feed glitches report `bid=5.20, ask=5.10` or `price=-0.10`. Corporate-action adjustments lag. Out-of-sequence prints survive in retail feeds. | A single negative or crossed quote can flip monotonicity/convexity for an entire slice. |
| **Corporate actions & expiries** | Special dividends, splits, and the expiry roll itself create discontinuities. Very near-expiry options (`T < 7d`) have gamma/vega that explodes, making IV inversion numerically unstable (`vega -> 0`). | Near-expiry inversion returns `None` or an absurd vol; deep ITM/OTM wings have almost no time value, so `sigma` is unidentifiable. |

**Design consequence** — quoting noise is not random measurement error you can average away. It is *structural* (missing sides, crossed books, zero prints) and *regime-dependent* (illiquid wings, near expiry). The only honest response is to **reject** what is unpriceable and **audit** what was rejected, not to silently smooth over it.

> **Project.md §2 — Quote Cleaning** enumerates the same target list: negative prices, zero/missing `bid`/`ask`, crossed markets, wide spreads, stale contracts, illiquid low-OI contracts, deep ITM/OTM, near-expiry, and intrinsic-value violations. Everything below is the implementation of that list (`docs/Project.md:160-174`).

---

## 2. Two-Layer Defense

```
yfinance DataFrame (calls / puts per expiry)
        │
        ▼  Layer 1 — Pre-ingestion quality filter
        │          src/arbfree_vol/data/quality.py:filter_option_chain()
        │          operates on raw DataFrames BEFORE any Quote exists
        │          DropRecord audit trail
        ▼
   Quote / ExpirySlice / VolSurface construction
        │  src/arbfree_vol/ingestion/_common.py:build_slice()
        │  src/arbfree_vol/ingestion/yahoo.py:fetch_chain()  (live)
        │  src/arbfree_vol/ingestion/loader.py:load_chain_csv() (CSV)
        ▼  Layer 2 — Cleaning layer
        │          src/arbfree_vol/ingestion/cleaning.py:clean_quotes()
        │          operates on VolSurface Quotes AFTER construction
        │          RejectionRecord audit trail
        ▼
   VolSurface → detect_with_forward() → repair()
```

**Why two layers?** The quality filter can see fields that the `Quote` model discards (`openInterest`, `volume`) and can reject a strike *before* it pollutes the surface. The cleaning layer can see fields the DataFrame cannot (`expiry_time`, `spot`, `log-moneyness`, intrinsic bounds) and enforces no-arbitrage preconditions. They are complementary, not redundant.

---

## 3. Layer 1 — Pre-Ingestion Quality Filter

### 3.1 What it is

`filter_option_chain()` in `src/arbfree_vol/data/quality.py:191` is called inside `build_slice()` for each expiry DataFrame *before* `Quote` objects are built. It is the first gate data passes through when `fetch_chain()` pulls from yfinance.

It enforces **two** thresholds via `DataQualityConfig` (`src/arbfree_vol/data/quality.py:24`):

| Field | Config key | Default | Unit | Enforced? |
|---|---|---|---|---|
| Open interest | `min_open_interest` | `10` | contracts | **Yes** — `oi < 10` → drop |
| Bid-ask spread | `max_bid_ask_pct` | `50.0` | % of mid (`(ask-bid)/mid * 100`) | **Yes** — `spread > 50%` → drop |
| Volume | — | — | contracts | **No** — recorded in `DropRecord.volume` for diagnostics only |

```python
from arbfree_vol.data.quality import DataQualityConfig, filter_option_chain

cfg = DataQualityConfig(min_open_interest=10, max_bid_ask_pct=50.0)
filtered_df, drops = filter_option_chain(df, expiry="2026-08-15", config=cfg)
# drops: list[DropRecord] — one per rejected strike
```

### 3.2 Why volume is NOT a filter

Daily per-strike `volume=0` is normal. Market-maker quotes away from ATM are legitimate two-sided markets that simply did not trade today. Open interest (outstanding contracts) and bid-ask width are the reliable liquidity signals; volume is not. Filtering on `volume` would drop good strikes across the wings and degrade the smile where you need it most. This is documented in the module docstring (`src/arbfree_vol/data/quality.py:8`) and in `docs/issues.md` Issue #15 follow-up.

Volume is still **recorded** in every `DropRecord` so audits can correlate drops with trading activity.

### 3.3 Missing vs. observed zero

A retail feed can fail in two indistinguishable-looking ways: `openInterest=0` (genuinely no open interest) and `openInterest` column absent / `None` / `NaN` / `pd.NA` (provider omitted the field). Treating the second as the first would mislabel an ingestion failure as "illiquid strike."

The filter distinguishes them explicitly (`src/arbfree_vol/data/quality.py:63-78, 166-188`):

* `_is_missing()` detects `None`, `NaN`, `pd.NA` via `pd.isna()`.
* `_extract_int_field()` / `_extract_float_field()` use `row.get(key)` with **no zero default** — an absent column returns `None`, which is flagged as missing.
* A missing `open_interest` is dropped with reason `OI=missing<10` and `missing_fields=("open_interest",)`.
* A missing `bid` or `ask` (one or both sides) is dropped with reason `spread=missing (missing: bid)` — because the mid would be fabricated (`missing bid -> +200%` spread if you naively used the one-sided price). Both-missing (the "N1 no-quote" strike) is the canonical target.
* A missing `volume` is recorded in `missing_fields` only — never a criterion.

This is the provenance guarantee: a mass drop caused by the provider omitting a column is never misreported as a liquidity filter.

### 3.4 Where it runs

* **Live path** — `src/arbfree_vol/ingestion/yahoo.py:35-77` (`fetch_chain()`): iterates expiries from `ticker.option_chain()`, calls `build_slice()` which internally calls `filter_option_chain()`. Returns a 3-tuple `(surface, rejected, quality_drops)`.
* **CSV path** — `src/arbfree_vol/ingestion/loader.py:54` (`load_chain_csv()`): CSVs have no `openInterest`/`volume` columns at this schema, so the CSV path does **not** apply the DataFrame quality filter; it relies entirely on Layer 2. Add OI columns to the CSV schema if you want Layer 1 there.

---

## 4. Layer 2 — The 8 Cleaning Rejection Rules

### 4.1 Overview

`clean_quotes()` in `src/arbfree_vol/ingestion/cleaning.py:168` applies 8 rejection rules to every `Quote` in an `ExpirySlice`. Each rule is a standalone `_check_*()` function that returns a `RejectionRecord` on failure or `None` on pass. The first rule that fires wins — subsequent rules are not evaluated for that quote (fail-fast, auditable). Kept quotes and rejected records are returned as `(kept, rejected)`.

Thresholds are **function arguments with defaults**, not inlined constants — the No-Hardcoding Rule from `docs/AGENTS.md` (`src/arbfree_vol/ingestion/cleaning.py:149` is the canonical example).

| # | `RejectionRule` | Check function | Threshold default | Operates on |
|---|---|---|---|---|
| 1 | `negative_price` | `_check_negative_price` | — (any `< 0`) | `price`, `bid`, `ask` |
| 2 | `zero_price` | `_check_zero_price` | — (`price == 0`) | `price` |
| 3 | `zero_bid_or_ask` | `_check_zero_bid_or_ask` | — (`None` or `== 0`) | `bid`, `ask` |
| 4 | `crossed_market` | `_check_crossed_market` | — (`bid > ask`) | `bid` vs `ask` |
| 5 | `wide_spread` | `_check_wide_spread` | `max_spread_ratio=0.5` (50%) | `(ask-bid)/mid` |
| 6 | `near_expiry` | `_check_near_expiry` | `min_T=7/365 ≈ 0.01918y` (7 days) | `sl.expiry_time` |
| 7 | `intrinsic_violation` | `_check_intrinsic_violation` | `1e-6` tolerance | `price` vs `max(0, S-K)` / `max(0, K-S)` |
| 8 | `deep_moneyness` | `_check_deep_moneyness` | `max_log_moneyness=1.5` | `|log(K/S)|` |

Threshold summary (what an interviewer wants on one slide):

| Parameter | Default | Meaning | Override |
|---|---|---|---|
| `min_T` | `7/365` | Minimum `T` in years (7 calendar days) | `clean_quotes(sl, spot, min_T=...)` |
| `max_spread_ratio` | `0.5` | Max `(ask-bid)/mid` as fraction (50%) | `clean_quotes(..., max_spread_ratio=...)` |
| `max_log_moneyness` | `1.5` | Max `|log(K/spot)|` (~78%–448% of spot) | `clean_quotes(..., max_log_moneyness=...)` |
| Intrinsic tolerance | `1e-6` | Numerical slack below intrinsic | Fixed (not parameterized) |

### 4.2 Rule-by-rule reference

#### Rule 1 — `negative_price` (`src/arbfree_vol/ingestion/cleaning.py:42`)

* **Condition:** `price < 0` or `bid < 0` or `ask < 0` (any side negative).
* **Threshold:** none — any negative is invalid.
* **Why it matters:** Option prices are non-negative by arbitrage. A negative print is a feed error or a corporate-action adjustment that arrived before the underlying price did. Inverting it would require `sigma` imaginary. Downstream, `implied_vol()` would return `None` and `slice_total_variance()` would silently drop the point — but the surface would carry a hole with no audit trail. Rejecting here makes the hole *visible*.
* **Example:** `Quote(strike=700, price=-0.05, bid=3.2, ask=3.5)` → `RejectionRecord(rule=NEGATIVE_PRICE, detail="price=-0.05")`.

#### Rule 2 — `zero_price` (`src/arbfree_vol/ingestion/cleaning.py:35`)

* **Condition:** `price == 0` exactly.
* **Threshold:** none — exact zero only (negative zero is Rule 1).
* **Why it matters:** `price=0` makes the IV solver degenerate — `vega` is undefined at the boundary and Brent/Newton have no bracket (`intrinsic=0` but `price=0` is not interior). The solver would return `None`; again, the failure would be silent. This rule fails fast and preserves the quote for the audit log.
* **Example:** `Quote(strike=900, price=0, bid=0.01, ask=0.05)` → `RejectionRecord(rule=ZERO_PRICE, detail="price=0")`. Note `bid=0.01` is not itself zero — Rule 3 handles the sides.

#### Rule 3 — `zero_bid_or_ask` (`src/arbfree_vol/ingestion/cleaning.py:55`)

* **Condition:** `bid is None` or `ask is None` (missing side) **or** `bid == 0` or `ask == 0` (observed no-quote).
* **Threshold:** none — missing or exact zero.
* **Why it matters:** This is the "no two-sided market" gate. A missing side means the provider had no quote on that side; `price` can only come from `lastPrice`, which may be stale by hours. A side of exactly `0` is the exchange's own no-quote signal. In either case there is no valid `mid`, so any IV derived from it is fabricated.
* **Subtlety:** `None` and `0` are treated under the *same* rule but produce different `detail` strings (`"missing: bid"` vs `"bid=0.0, ask=1.3"`) so an audit can distinguish provider omission from an observed no-quote.
* **Example:** `Quote(strike=800, price=2.1, bid=None, ask=2.5)` → `RejectionRecord(rule=ZERO_BID_OR_ASK, detail="missing: bid")`.

#### Rule 4 — `crossed_market` (`src/arbfree_vol/ingestion/cleaning.py:75`)

* **Condition:** `bid > ask` (with both sides present).
* **Threshold:** none — any crossing.
* **Why it matters:** A crossed book (`bid` above `ask`) is a market-data error — it implies you could buy at `ask` and instantly sell at `bid` for a risk-free profit. No real book looks like this; it is an out-of-sequence print or a feed merge bug. Using its `mid` would invert a price that violates the most basic market invariant.
* **Guard:** returns `None` when either side is `None` — that case is already handled by Rule 3 and should not double-fire.
* **Example:** `Quote(strike=750, price=4.0, bid=4.2, ask=4.0)` → `RejectionRecord(rule=CROSSED_MARKET, detail="bid=4.2 > ask=4.0")`.

#### Rule 5 — `wide_spread` (`src/arbfree_vol/ingestion/cleaning.py:87`)

* **Condition:** `(ask - bid) / mid > max_spread_ratio`, where `mid=(bid+ask)/2`.
* **Threshold default:** `max_spread_ratio=0.5` (i.e. spread exceeds 50% of mid).
* **Why it matters:** A 50%+ spread means the market maker has almost no conviction in the price — typical in the far wings where inventory risk is high. The `mid` is then a noisy midpoint of a very wide interval; its IV can swing wildly with a 1-tick move in either side. SVI fitted to such points will chase noise and produce butterfly violations at the wings. Rejecting here protects the smile's wings.
* **Guards:** skips when either side is `None`, `<=0`, or `mid <=0` — those cases are caught by Rules 3–4.
* **Example:** `Quote(strike=950, price=0.40, bid=0.20, ask=0.60)` → `mid=0.40`, `spread/mid=1.0 > 0.5` → `RejectionRecord(rule=WIDE_SPREAD, detail="spread/mid=1.0000 > 0.5")`.
* **Tuning:** tighten to `0.30` for very liquid underlyings (SPY, SPX); loosen to `0.80` when you deliberately want wing coverage and accept noisier IVs.

#### Rule 6 — `near_expiry` (`src/arbfree_vol/ingestion/cleaning.py:107`)

* **Condition:** `sl.expiry_time < min_T` — the *slice's* time-to-expiry, not the quote's.
* **Threshold default:** `min_T=7/365 ≈ 0.01918` years (7 calendar days).
* **Why it matters:** For `T -> 0`, Black-Scholes `vega -> 0` and `theta -> ±∞`. The IV inversion becomes ill-conditioned: a $0.01 price error maps to an enormous `sigma` error, and the solver may not converge at all. Gatheral's SVI literature and the project's own Issue #3 note that `T < 0.05y` already produces steep-wing butterfly failures even *after* calibration. Cutting at 7 days is a pragmatic floor; `loader.py:32` and `yahoo.py:143` use `days/365.0` (ACT/365) for `T`, so this threshold is in the same day-count convention.
* **Note:** this rejects the *entire slice* quote-by-quote — every quote in a near-expiry slice is rejected individually (not the slice as a whole), so the audit log shows `N` records and `loader.py:91` drops the slice only if `kept` is empty.
* **Example:** slice with `expiry_time=0.0137` (5 days) → every quote → `RejectionRecord(rule=NEAR_EXPIRY, detail="T=0.0137 < 0.01918")`.

#### Rule 7 — `intrinsic_violation` (`src/arbfree_vol/ingestion/cleaning.py:116`)

* **Condition:** `price < intrinsic - 1e-6`, where `intrinsic = max(0, S-K)` for calls, `max(0, K-S)` for puts. Uses undiscounted intrinsic (no `exp(-rT)` / `exp(-qT)`) — deliberately coarse, because the forward curve is not yet available at cleaning time (same rationale as Rule 8).
* **Threshold:** `1e-6` slack for floating-point rounding; not parameterized.
* **Why it matters:** `price < intrinsic` is a static arbitrage — you could buy the option and exercise immediately for a risk-free profit. A quote violating this is either stale (underlying moved, quote did not) or a bad print. The IV solver would have no root in the bracket `[intrinsic, S]` and would return `None`; again, silent failure without this gate.
* **Discounting note:** the check uses *undiscounted* intrinsic (`S-K`, not `S*exp(-qT)-K*exp(-rT)`). This is slightly conservative — it may *pass* a quote that violates discounted parity but not undiscounted. The precise parity check (with `r`, `q`, and forward) is deferred to `arbitrage/quote_detect.py:_check_parity`, which runs *after* cleaning with per-slice rates. The cleaning rule is a coarse, rate-free guard.
* **Example:** `spot=760, Quote(strike=700, option_type=CALL, price=50.0)` → `intrinsic=60.0`, `50 < 60 - 1e-6` → `RejectionRecord(rule=INTRINSIC_VIOLATION, detail="call price=50.0000 < intrinsic=60.0000")`.

#### Rule 8 — `deep_moneyness` (`src/arbfree_vol/ingestion/cleaning.py:140`)

* **Condition:** `|log(K / spot)| > max_k`.
* **Threshold default:** `max_log_moneyness=1.5` in `|k|`.
* **Why it matters:** At `|k|=1.5`, `K/spot ≈ 0.22` or `4.48` — far beyond any liquid strike. Deep ITM options are essentially forward contracts (time value ≈ 0, vega ≈ 0); deep OTM options are lottery tickets with `price ≈ 0` and no vega. In both cases IV is unidentifiable — the price is insensitive to `sigma`, so inversion amplifies noise. SVI fitted to such points will extrapolate wildly. Cutting at `|k|=1.5` keeps the smile in the region where time value is meaningful.
* **Moneyness divergence — intentional:** this guard uses `log(K/spot)` (spot-based) while the downstream pipeline uses `log(K/F)` with `F=S*exp((r-q)T)` (forward-based, `src/arbfree_vol/svi/data.py:_forward_price`). The forward curve is *estimated after cleaning* from surviving quotes (`repair/fwd_curve.py:estimate_forward_curve`, median put-call parity), so it is not available here. The docstring calls this out explicitly: the cleaning check is a coarse, rate-free guard; the precise forward-based moneyness is computed later. A small disagreement at the boundary (e.g. `r-q ≈ 3%`, `dT ≈ 1y` shifts `k` by ~0.03) is acceptable — it just means a quote near `|k|=1.5` may be borderline, which is the right place for a guard to be conservative.
* **Example:** `spot=760, Quote(strike=200, ...)` → `k=log(200/760)≈-1.335` → kept (just inside). `strike=100` → `k≈-2.028` → `RejectionRecord(rule=DEEP_MONEYNESS, detail="|k|=2.0280 > 1.5")`.

### 4.3 Evaluation order

Rules are evaluated in the order listed above (`src/arbfree_vol/ingestion/cleaning.py:185-194`). The first rejection encountered is recorded and the remaining checks are skipped for that quote. Order matters for audit clarity:

* Structural impossibilities (`negative_price`, `zero_price`) fire first — no point checking spreads on a negative price.
* Missing-market checks (`zero_bid_or_ask`, `crossed_market`) fire next — no point computing `spread/mid` on a crossed or one-sided book.
* Economic guards (`wide_spread`, `near_expiry`, `intrinsic_violation`, `deep_moneyness`) fire last — they assume a structurally valid quote.

---

## 5. Cleaning Layer vs. Quality Filter — The Distinction

| Dimension | Quality filter (`src/arbfree_vol/data/quality.py`) | Cleaning layer (`src/arbfree_vol/ingestion/cleaning.py`) |
|---|---|---|
| **Input** | Raw `DataFrame` per expiry (yfinance `calls`/`puts`) | `ExpirySlice` / `Quote` objects (already constructed) |
| **When** | *Before* `Quote` construction, inside `build_slice()` | *After* `Quote`/`ExpirySlice` construction |
| **Sees** | `strike`, `openInterest`, `volume`, `bid`, `ask` | `strike`, `price`, `bid`, `ask`, `spot`, `expiry_time`, `option_type` |
| **Does not see** | `spot`, `T`, `option_type` semantics | `openInterest`, `volume` (already discarded) |
| **Thresholds** | `min_open_interest=10`, `max_bid_ask_pct=50%` | `min_T=7/365`, `max_spread_ratio=0.5`, `max_log_moneyness=1.5`, `1e-6` intrinsic slack |
| **Volume** | Recorded, never enforced | Not applicable (not in model) |
| **Audit type** | `DropRecord` (`strike`, `expiry`, `reason`, `open_interest`, `volume`, `bid_ask_pct`, `missing_fields`) | `RejectionRecord` (`quote`, `rule: RejectionRule`, `detail: str`) |
| **CSV path** | Not applied (no OI/volume columns) | Applied via `load_chain_csv(clean=True)` |
| **Disable** | `fetch_chain(disable_quality_filter=True)` | `load_chain_csv(clean=False)` or call `clean_quotes` selectively |

**Interview line:** "The quality filter asks *is this strike liquid enough to quote?* using market-microstructure fields. The cleaning layer asks *is this quote structurally priceable?* using arbitrage preconditions. They run at different abstraction levels and produce different audit trails — conflating them would lose either liquidity context or price-structure context."

---

## 6. Audit Trails — Preserving What Was Rejected

Both layers are **report, not raise** — they never silently discard data. Every rejection is preserved with a reason.

### 6.1 `DropRecord` — quality filter (`src/arbfree_vol/data/quality.py:41`)

```python
@dataclass(frozen=True, slots=True)
class DropRecord:
    strike: float
    expiry: str          # e.g. "2026-08-15"
    reason: str          # e.g. "OI=3<10; spread=62.5%>50.0%"
    open_interest: int
    volume: int          # diagnostic only
    bid_ask_pct: float
    missing_fields: tuple[str, ...]  # e.g. ("open_interest",) or ("bid",)
```

* One record per rejected DataFrame row (strike).
* `reason` concatenates all failing thresholds with `"; "` — a strike failing both OI and spread gets one record with both reasons.
* `missing_fields` distinguishes provider omission from observed zero (see §3.3).
* Returned as `list[DropRecord]` (third element of `fetch_chain()`).

### 6.2 `RejectionRecord` — cleaning layer (`src/arbfree_vol/ingestion/cleaning.py:27`)

```python
@dataclass(frozen=True, slots=True)
class RejectionRecord:
    quote: Quote              # the full Quote that was rejected
    rule: RejectionRule       # enum — machine-readable
    detail: str               # e.g. "spread/mid=0.6234 > 0.5"
```

* One record per rejected `Quote` — the `quote` field preserves the original `strike`, `price`, `bid`, `ask`, `option_type` for forensics.
* `rule` is an enum (`RejectionRule`) for programmatic grouping; `detail` is a human-readable string with the offending value and threshold.
* Returned as `list[RejectionRecord]` (second element of `fetch_chain()` / `load_chain_csv()`).

### 6.3 Using the trails

```python
from collections import Counter

# Quality filter — how much was dropped and why
surface, rejected, quality_drops = fetch_chain("SPY", max_expiries=5)
print(f"quality drops: {len(quality_drops)}")
print(Counter(d.reason for d in quality_drops))
# e.g. Counter({"OI=3<10": 812, "spread=62.5%>50.0%": 19})

# Cleaning layer — which structural rule fired
print(Counter(r.rule.value for r in rejected))
# e.g. Counter({"wide_spread": 42, "deep_moneyness": 18, "near_expiry": 0})

# Drill into missing-field vs observed-zero
missing_oi = [d for d in quality_drops if "open_interest" in d.missing_fields]
print(f"dropped due to missing OI column: {len(missing_oi)}")
```

Both trails are consumed by the yfinance demo (`demo/yfinance/yfinance_demo.py`) which prints drop counts and breakdowns, and by the Issue #15 audit script (`scripts/audit_theta_dip_data_quality.py`) which correlates fallback slices with pre-filter OI/spread metrics.

---

## 7. Handling Illiquid Contracts, Wide Spreads, Deep Moneyness, Near-Expiry

These four cases share a design principle — **thresholds are function arguments with defaults, not magic numbers** (`docs/AGENTS.md` — No-Hardcoding Rule, citing `cleaning.py:clean_quotes()` as the canonical pattern).

| Challenge | Quality-filter threshold | Cleaning-layer threshold | Tuning guidance |
|---|---|---|---|
| **Illiquid / thin OI** | `DataQualityConfig.min_open_interest=10` | — (no OI at this layer) | Raise to `50–100` for very liquid names; lower to `1` when you need wing coverage and accept noisier fits. `docs/issues.md` Issue #15 showed fallback expiries had median OI 297 vs 794 for good expiries — the default `10` is intentionally permissive; most wing drops come from `OI < 10` on far OTM strikes. |
| **Wide spreads** | `DataQualityConfig.max_bid_ask_pct=50.0` (%) | `max_spread_ratio=0.5` (fraction) — same 50% in different units | Both default to 50%. Tighten to 30% for SPY/SPX; loosen to 80% for single-names or when deliberately studying illiquid wings. The cleaning check guards against `bid<=0` / `mid<=0` before computing the ratio, so crossed/missing quotes do not double-fire here. |
| **Deep moneyness** | — | `max_log_moneyness=1.5` | `|k|=1.5` ≈ 22%–448% of spot. Widen to `2.0` for long-dated surfaces where you want more wing; tighten to `1.0` for short-dated where wings are unreliable. Uses `log(K/spot)` spot-based — see §4.2 Rule 8 for the deliberate divergence from forward-based `k`. |
| **Near-expiry** | — (no `T` at DataFrame layer) | `min_T=7/365` | Raise to `14/365` or `30/365` when calibrating eSSVI — `docs/issues.md` Issue #3 notes `T < 0.05y` already produces SVI butterfly violations even after cleaning. The yfinance path also has `fetch_chain(min_T_years=...)` which skips expiries *before* building slices — a coarser pre-filter that complements this per-quote check. |

**Stale / zero volume** is handled implicitly, not by a threshold:

* **Stale quotes** — no explicit "staleness" signal exists in the yfinance feed. The project uses *proxies*: near-expiry (`T < 7d`) where staleness is most damaging, wide spreads where market makers have withdrawn, and `using mid not lastPrice` (`fetch_chain` docstring, `src/arbfree_vol/ingestion/yahoo.py:44` — `build_slice` computes `price=(bid+ask)/2`, never `lastPrice`). `lastPrice` is intentionally ignored because it may be hours stale.
* **Zero volume** — deliberately *not* filtered (see §3.2). The quality filter records it; the cleaning layer does not see it. Filtering on `volume` would discard legitimate away-from-ATM quotes that simply did not trade today.

---

## 8. How to Configure

### 8.1 Defaults

```python
# Quality filter defaults
DataQualityConfig(min_open_interest=10, max_bid_ask_pct=50.0)

# Cleaning defaults
clean_quotes(sl, spot, min_T=7/365, max_spread_ratio=0.5, max_log_moneyness=1.5)
```

All defaults are chosen to be **permissive** — they remove only the clearly unpriceable tail, not the noisy-but-usable middle. Research (Issue #15) showed that tightening them further had mixed effects and that snapshot-to-snapshot variation dominates any single threshold choice.

### 8.2 Per-call overrides

```python
from arbfree_vol.data.quality import DataQualityConfig
from arbfree_vol.ingestion.cleaning import clean_quotes

# Stricter quality filter for a liquid-name study
strict = DataQualityConfig(min_open_interest=100, max_bid_ask_pct=30.0)
surface, rejected, quality_drops = fetch_chain("SPY", quality_config=strict)

# Looser cleaning for wing research
from arbfree_vol.models.surface import ExpirySlice, Quote
kept, rejected = clean_quotes(sl, spot, max_spread_ratio=0.80, max_log_moneyness=2.0)

# CSV path — cleaning is on by default
from arbfree_vol.ingestion.loader import load_chain_csv
surface, rejected = load_chain_csv("data/spy_chain.csv", spot=754.0)  # clean=True default
surface_raw, _ = load_chain_csv("data/spy_chain.csv", spot=754.0, clean=False)
```

### 8.3 YAML / config-file usage

`DataQualityConfig` is a plain dataclass — it composes with any config loader. No YAML schema is enforced by the library; the recommended pattern is:

```yaml
# config/data_quality.yaml
data_quality:
  min_open_interest: 50
  max_bid_ask_pct: 30.0
cleaning:
  min_T_days: 14
  max_spread_ratio: 0.30
  max_log_moneyness: 1.2
```

```python
import yaml
from arbfree_vol.data.quality import DataQualityConfig
from pathlib import Path

cfg_raw = yaml.safe_load(Path("config/data_quality.yaml").read_text())
qc = DataQualityConfig(
    min_open_interest=cfg_raw["data_quality"]["min_open_interest"],
    max_bid_ask_pct=cfg_raw["data_quality"]["max_bid_ask_pct"],
)
surface, rejected, drops = fetch_chain("SPY", quality_config=qc)
# Per-slice cleaning thresholds are passed directly to clean_quotes()
# or via build_slice() — no global YAML key for them; they are call-site args.
```

### 8.4 Raw access — disabling filters

For audits and for reproducing Issue #15 before/after comparisons, you need truly unfiltered data. There are two independent switches:

| Path | Flag | Effect |
|---|---|---|
| `fetch_chain()` (live) | `disable_quality_filter=True` | Skips `filter_option_chain()` entirely; raw yfinance DataFrames are returned. **This is the ONLY way to get truly unfiltered live data** — passing `quality_config=None` with `disable_quality_filter=False` still applies `DataQualityConfig()` defaults (`src/arbfree_vol/ingestion/yahoo.py:39-77`). |
| `load_chain_csv()` (CSV) | `clean=False` | Skips `clean_quotes()` entirely; every CSV row becomes a `Quote` (`src/arbfree_vol/ingestion/loader.py:88`). |

```python
# True baseline for an audit — no quality filter, no cleaning
surface_raw, rejected_raw, drops_raw = fetch_chain(
    "SPY", disable_quality_filter=True
)
# drops_raw == [] — nothing was filtered at Layer 1
# rejected_raw still contains Layer 2 rejections unless you also bypass build_slice()

# Audit script pattern (scripts/audit_theta_dip_data_quality.py)
# Runs twice on the same calendar date: filter OFF vs filter ON, then compares.
```

> **Calendar-date caveat** (`docs/issues.md` Issue #15): live fallback/drop counts are snapshot-in-time. Expiries roll daily and quotes update in real time. Do not compare counts across calendar dates — run OFF vs ON on the *same* fetch.

---

## 9. Interview Cheat Sheet

**"Why not just use `lastPrice`?"**
`lastPrice` is the last *trade*, which on illiquid strikes may be hours or days old while the underlying has moved. `mid=(bid+ask)/2` is the market maker's current two-sided commitment. The project never uses `lastPrice` — `build_slice()` computes `price` from `bid`/`ask` only.

**"Why not filter on volume?"**
Daily per-strike `volume=0` is normal for legitimate quotes away from ATM. Filtering on it would discard the wings you need for the smile. OI and spread are the reliable liquidity signals; volume is diagnostic, not decisive.

**"Why two layers instead of one?"**
They see disjoint fields. The DataFrame layer sees `openInterest`/`volume` but not `spot`/`T`; the `Quote` layer sees `spot`/`T`/intrinsic but not OI. Merging them would require threading raw DataFrame fields through the `Quote` model (Pydantic boundary pollution) or recomputing `T`/`moneyness` inside the DataFrame filter (forward curve not yet available). Separation keeps each layer's abstraction clean.

**"What happens to rejected quotes?"**
Nothing is silently dropped. Every rejection is preserved as a `DropRecord` (Layer 1) or `RejectionRecord` (Layer 2) with the reason and the offending value. The yfinance demo and `RepairReport` surface these counts; `scripts/audit_theta_dip_data_quality.py` correlates them with calibration fallbacks.

**"How do you choose thresholds?"**
Defaults are permissive and documented as function arguments, not magic numbers. The audit in Issue #15 measured the actual OI/spread distribution (median fallback OI 297 vs 794 for good expiries) and showed that tightening OI from 10 to 50 primarily affects wing strikes. The right threshold depends on the underlying's liquidity regime — SPY tolerates stricter filters than a single-name.

**"What about corporate actions / splits?"**
yfinance adjusts `strike`/`spot` for splits but special dividends can lag. A dividend overstatement makes calls look cheap (intrinsic violation) and puts look rich — the intrinsic check catches the cheap calls, but a dividend *under*statement would pass cleaning and surface as a parity violation in `detect_with_forward()`. That is intentional: cleaning is coarse; precise parity with per-slice forwards is the detector's job.

---

## 10. Data Flow Summary

```
fetch_chain("SPY")                        load_chain_csv("chain.csv")
      │                                           │
      ├─ ticker.option_chain(expiry)              ├─ csv.DictReader
      │   calls/puts DataFrames                   │   Quote(price, bid, ask)
      │         │                                 │         │
      │         ▼                                 │         ▼
      │  filter_option_chain(df, expiry, cfg)     │   ExpirySlice(quotes)
      │  DropRecord per rejected strike           │         │
      │  (OI<10, spread>50%, missing sides)       │         ▼
      │         │                                 │  clean_quotes(sl, spot)
      │         ▼                                 │  RejectionRecord per bad quote
      │  build_slice(calls, puts, exp_str)        │  (8 rules, §4.2)
      │  Quote(price=mid, bid, ask)               │         │
      │         │                                 │         ▼
      └─────────┼─────────────────────────────────┘
                ▼
           VolSurface(spot, r, q, slices)
                │
                ▼
         detect_with_forward()  ←  estimate_forward_curve() (median parity)
                │
                ▼
            repair()  →  FittedSlice / RepairReport
```

---

## 11. References

* **Source of truth — quality filter:** `src/arbfree_vol/data/quality.py` — `DataQualityConfig`, `DropRecord`, `filter_option_chain()`, `_evaluate_row()`, `_is_missing()`.
* **Source of truth — cleaning rules:** `src/arbfree_vol/ingestion/cleaning.py` — `RejectionRule`, `RejectionRecord`, `_check_*()`, `clean_quotes()`.
* **Live ingestion:** `src/arbfree_vol/ingestion/yahoo.py:35` — `fetch_chain()`; `src/arbfree_vol/ingestion/_common.py:build_slice()`.
* **CSV ingestion:** `src/arbfree_vol/ingestion/loader.py:54` — `load_chain_csv()`.
* **Project spec — Quote Cleaning:** `docs/Project.md:160-174`.
* **Conventions — No-Hardcoding, thresholds as args:** `docs/AGENTS.md` — No-Hardcoding Rule.
* **Data quality audit & historical results:** `docs/issues.md` Issue #15 — Data quality audit, Corrected audit approach, Data source comparison, Determinism check, Calendar date caveat.
* **Forward curve (downstream):** `src/arbfree_vol/repair/fwd_curve.py:estimate_forward_curve()` — median-based, used by `detect_with_forward()`.
* **Audit consumers:** `demo/yfinance/yfinance_demo.py`, `scripts/audit_theta_dip_data_quality.py`, `tests/test_cleaning.py`, `tests/test_data_quality.py`.
