# ADR-0001: Constrained SVI calibration with pluggable SABR smile model

- Status: Accepted
- Date: 2026-08-13 (reflects code state at commit `4b0363d`)

## Context

The repair pipeline originally used unconstrained SVI calibration, which
produced butterfly-arbitrage violations in fitted slices, and the smile-model
layer supported only two pluggable models (SVI, eSSVI).

## Decision

- `calibrate_constrained()` in `src/arbfree_vol/svi/calibration.py` augments
  the SVI residual with a butterfly-arbitrage penalty
  `sqrt(arb_penalty) * sqrt(max(-g(k), 0))` over a k-grid, a min-variance
  term, and a calendar penalty that forces `w_current(k) >= w_prev(k)` on the
  k-grid across expiries.
- The repair pipeline (`repair()` in `src/arbfree_vol/repair/engine.py`)
  dispatches to one of three paths — `_repair_svi`, `_repair_essvi`,
  `_repair_sabr` — selected by the mutually exclusive `use_ssvi` / `use_sabr`
  flags:
  - **raw SVI** (default): per-slice constrained calibration.
  - **eSSVI** (`use_ssvi=True`): sequential H&M calendar-arb-free fit with
    per-slice `fallback_slices` and `failed_slices` tracking.
  - **SABR** (`use_sabr=True`): B-spline term-structure fit mapped to raw SVI.
- SABR (Hagan et al. 2002) is a third pluggable smile model with a pydantic
  boundary type, an ATM closed-form limit, and the full Hagan asymptotic
  formula. Its `to_raw_svi_params()` adapter in `src/arbfree_vol/sabr/model.py`
  samples SABR on a center-weighted k-grid and fits raw SVI, so the downstream
  pipeline stays SVI-parameterized. The engine imports it as
  `sabr_to_raw_svi_params` (distinct from the eSSVI→SVI `to_raw_svi_params` in
  `src/arbfree_vol/ssvi/model.py`).
- Fitted-surface types (`FittedSlice`, `FittedSSVISlice`, `FittedSABRSlice`)
  live in `src/arbfree_vol/models/fitted.py`, moved there to break the
  pricing → repair dependency.

## Consequences

- Fewer butterfly violations by construction on the SVI path; calendar
  arbitrage is handled per-path (SVI via the calendar penalty, eSSVI via H&M).
- SABR slices surface as native `FittedSABRSlice` plus a raw-SVI mapping for
  downstream SVI-based consumers (plots, detection).
- The SABR path verifies the mapped slices via grid-based `detect_svi_surface`
  rather than an in-fit butterfly penalty.

## Supersedes

This is the authoritative refresh of the repowise-inferred decision
"Add SABR model and switch to constrained SVI calibration" (id
`99fcc2f2c66e4553924daf40e093ee7e`), whose text predates the calendar
penalty, the three-path dispatch, the center-weighted mapping grid, and the
`models/fitted.py` move.
