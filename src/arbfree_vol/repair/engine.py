"""Model-agnostic repair pipeline.

``repair`` runs the quote-rejection / forward-estimation / refit
sequence and delegates the actual per-model fitting to a
``RepairStrategy`` selected by ``get_strategy``.  The pipeline itself
has no knowledge of any model API: all SVI / eSSVI / SABR imports and
fitting logic live in ``arbfree_vol.repair.strategies``, so adding a
model family means adding one strategy module there and nothing else.

Churn note (top-1% file): ``engine.py`` collects the cross-cutting
bookkeeping contracts that every repair path must honor — the
``failed_slices`` chronological-ordering + dedup contract
(``_consolidate_failures``), the no-forward expiry handling, and the
``_PathFitResult`` field surface.  When a contract changes, the edit
lands here and in the strategy modules together; keep each contract's
logic in its named step helper (``_detect_violations``, ``_reject_quotes``,
``_build_cleaned_surface``, ``_estimate_forward``, ``_fit_slices``,
``_verify_remaining``, ``_build_metrics``) so the pipeline body stays a
flat 1-7 read.
"""

from arbfree_vol.models.surface import VolSurface, ExpirySlice
from arbfree_vol.models.option import OptionType
from arbfree_vol.arbitrage.report import ArbitrageReport, ArbitrageViolation
from arbfree_vol.arbitrage.quote_detect import detect_with_forward
from arbfree_vol.arbitrage.svi_detect import detect_svi_surface
from arbfree_vol.models.fitted import FittedSlice
from arbfree_vol.repair.report import (
    RejectedQuote,
    RepairMetrics,
    RepairReport,
)
from arbfree_vol.forward import estimate_forward_curve, populate_per_slice_r
from arbfree_vol.repair.strategies import RepairStrategy, get_strategy, _PathFitResult


def _build_rejection_set(
    violations,
) -> tuple[set[tuple[float, float, OptionType]], list[RejectedQuote]]:
    """Collect all offending quotes from violations into a deduplicated set.

    Returns (identity_set, rejected_quote_list) where the set is used for
    fast lookup and the list preserves the rejection reason.
    """
    seen: set[tuple[float, float, OptionType]]= set()
    rejected: list[RejectedQuote]= []

    for v in violations:
        for oq in v.offending:
            key= (oq.strike, oq.expiry_time, oq.option_type)
            if key not in seen:
                seen.add(key)
                rejected.append(
                    RejectedQuote(
                        strike=oq.strike,
                        expiry_time=oq.expiry_time,
                        option_type=oq.option_type,
                        reason=v.kind,
                        magnitude=v.magnitude,
                    )
                )

    return seen, rejected


def _detect_violations(
    surface: VolSurface,
) -> tuple[list[ArbitrageViolation], int]:
    """Pipeline step 1: forward-aware arb detection on the raw surface.

    Returns ``(violations, count)``.
    """
    arb_report = detect_with_forward(surface)
    return arb_report.violations, len(arb_report.violations)


def _reject_quotes(
    violations,
) -> tuple[set[tuple[float, float, OptionType]], list[RejectedQuote]]:
    """Pipeline step 2: build the deduplicated rejection set."""
    return _build_rejection_set(violations)


def _build_cleaned_surface(surface: VolSurface, 
                           reject_set: set[tuple[float, float, OptionType]]
) -> VolSurface | None:
   
    """Remove all rejected quotes and drop empty slices."""
    cleaned: list[ExpirySlice]= []
    for sl in surface.slices:
        kept= []
        for q in sl.quotes:
            key= (q.strike, sl.expiry_time, q.option_type)
            if key not in reject_set:
                kept.append(q)
        if kept:
            cleaned.append(ExpirySlice(
                expiry_time=sl.expiry_time, quotes=kept,
                risk_free=sl.risk_free, div_yield=sl.div_yield,
            ))

    if not cleaned:
        return None
    return VolSurface(
        spot=surface.spot,
        risk_free=surface.risk_free,
        div_yield=surface.div_yield,
        slices=cleaned,
    )


def _estimate_forward(
    cleaned_surface: VolSurface | None,
) -> dict[float, float]:
    """Pipeline step 4: estimate the forward curve from survivors.

    Also populates each surviving slice's per-slice risk-free rate from
    the curve (a detection-time correction, never persisted back to the
    caller's surface).  Empty when no cleaned surface survives.
    """
    if cleaned_surface is None:
        return {}
    fwd_curve = estimate_forward_curve(cleaned_surface)
    populate_per_slice_r(cleaned_surface, fwd_curve)
    return fwd_curve


def _fit_slices(
    strategy: RepairStrategy,
    cleaned_surface: VolSurface | None,
    fwd_curve: dict[float, float],
) -> _PathFitResult:
    """Pipeline step 5: run the model strategy's per-slice fit.

    Returns an empty ``_PathFitResult`` when nothing survives rejection.
    """
    if cleaned_surface is None:
        return _PathFitResult(fitted=[], failed_slices=[])
    return strategy.fit(cleaned_surface, fwd_curve)


def _verify_remaining(fitted: list[FittedSlice]) -> ArbitrageReport:
    """Pipeline step 6: grid-check the fitted slices for violations.

    A fit that leaves grid-detectable violations is not feasible, even
    if the eSSVI H&M parameter check passed (the grid runs the raw-SVI
    mapping at discrete k, which can surface violations the parameter
    check misses).
    """
    if not fitted:
        return ArbitrageReport(violations=[])
    svi_slices = [(fs.expiry_time, fs.params) for fs in fitted]
    return detect_svi_surface(svi_slices)


def _build_metrics(
    *,
    n_rejected: int,
    n_total_quotes: int,
    n_slices_input: int,
    n_slices_fitted: int,
    n_violations_before: int,
    n_violations_after: int,
) -> RepairMetrics:
    """Pipeline step 7: assemble the repair metrics."""
    return RepairMetrics(
        n_rejected=n_rejected,
        n_total_quotes=n_total_quotes,
        n_slices_input=n_slices_input,
        n_slices_fitted=n_slices_fitted,
        n_violations_before=n_violations_before,
        n_violations_after=n_violations_after,
    )


def _consolidate_failures(failed_slices: list[float]) -> list[float]:
    """Sort + dedupe failed-slice expiries (the report ordering contract).

    The eSSVI path reassigns the list from the sequential fit result and
    then appends the no-forward expiries, which can produce
    non-chronological lists (a no-forward expiry sorting BEFORE a
    failed-fit expiry).  Every path's failures are merged, sorted by
    expiry and deduplicated before being stored in the report, so
    consumers may rely on chronological order.
    """
    return sorted(set(failed_slices))


def _run_pipeline(surface: VolSurface, strategy: RepairStrategy) -> RepairReport:
    n_total_quotes = sum(len(sl.quotes) for sl in surface.slices)
    n_slices_input = len(surface.slices)

    # step 1-2: detect violations on the raw surface, build the rejection set
    violations, n_violations_before = _detect_violations(surface)
    reject_set, rejected = _reject_quotes(violations)

    # step 3: build the cleaned surface
    cleaned_surface = _build_cleaned_surface(surface, reject_set)

    # step 4: estimate forward curve from survivors and populate per-slice r
    fwd_curve = _estimate_forward(cleaned_surface)

    # step 5: fit SVI (or eSSVI or SABR) on each cleaned slice
    path_result = _fit_slices(strategy, cleaned_surface, fwd_curve)
    fitted = path_result.fitted
    fitted_ssvi = path_result.fitted_ssvi
    fitted_sabr = path_result.fitted_sabr
    fallback_slices = path_result.fallback_slices
    failed_slices = path_result.failed_slices
    sabr_mapping_failed = path_result.sabr_mapping_failed
    repair_infeasible = path_result.repair_infeasible

    # step 6: detect remaining violations on the fitted surface
    remaining = _verify_remaining(fitted)
    if remaining.violations:
        repair_infeasible = True

    # step 7: metrics, then the failed_slices ordering contract
    metrics = _build_metrics(
        n_rejected=len(rejected),
        n_total_quotes=n_total_quotes,
        n_slices_input=n_slices_input,
        n_slices_fitted=len(fitted),
        n_violations_before=n_violations_before,
        n_violations_after=len(remaining.violations),
    )
    failed_slices = _consolidate_failures(failed_slices)

    return RepairReport(
        rejected=tuple(rejected),
        fitted_slices=tuple(fitted),
        fitted_ssvi_slices=tuple(fitted_ssvi),
        fitted_sabr_slices=tuple(fitted_sabr),
        remaining_violations=remaining,
        metrics=metrics,
        cleaned_surface=cleaned_surface,
        repair_infeasible=repair_infeasible,
        fallback_slices=fallback_slices,
        failed_slices=failed_slices,
        sabr_mapping_failed_slices=sabr_mapping_failed,
    )


def repair(surface: VolSurface, use_ssvi: bool= False, use_sabr: bool= False) -> RepairReport:
    """Repair a volatility surface by rejecting arb violating quotes,
    re-estimating the forward curve, and refitting SVI slices.

    Dispatches to one of three model-specific strategies in
    ``arbfree_vol.repair.strategies``: ``SVIStrategy`` (raw SVI),
    ``ESSVIStrategy`` (eSSVI), or ``SABRStrategy`` (SABR) — see those
    strategies for the per-model fitting details.

    ``RepairReport.failed_slices`` records expiries that could not be
    fitted at all — eSSVI slices where both the hard fit and the
    unconstrained fallback fail, raw-SVI slices whose constrained
    calibration raises, and (in every path) slices for which the forward
    curve has no estimate.  The no-forward case is logged with a warning
    and recorded, never silently skipped.  The list is SORTED by expiry
    and DEDUPLICATED before being stored in the report: the eSSVI path
    reassigns ``failed_slices`` from the sequential fit result and then
    appends the no-forward expiries, which can otherwise produce
    non-chronological lists (a no-forward expiry sorting BEFORE a
    failed-fit expiry).  Consumers may rely on chronological order.

    ``use_ssvi`` and ``use_sabr`` are mutually exclusive.
    """
    return _run_pipeline(surface, get_strategy(use_ssvi, use_sabr))
