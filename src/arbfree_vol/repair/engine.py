"""Model-agnostic repair pipeline.

``repair`` runs the quote-rejection / forward-estimation / refit
sequence and delegates the actual per-model fitting to a
``RepairStrategy`` selected by ``get_strategy``.  The pipeline itself
has no knowledge of any model API: all SVI / eSSVI / SABR imports and
fitting logic live in ``arbfree_vol.repair.strategies``, so adding a
model family means adding one strategy module there and nothing else.
"""

from arbfree_vol.models.surface import VolSurface, ExpirySlice
from arbfree_vol.models.option import OptionType
from arbfree_vol.arbitrage.report import ArbitrageReport
from arbfree_vol.arbitrage.quote_detect import detect_with_forward
from arbfree_vol.arbitrage.svi_detect import detect_svi_surface
from arbfree_vol.models.fitted import FittedSlice, FittedSSVISlice, FittedSABRSlice
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


def _run_pipeline(surface: VolSurface, strategy: RepairStrategy) -> RepairReport:
    n_total_quotes= sum(len(sl.quotes) for sl in surface.slices)
    n_slices_input= len(surface.slices)

    # step 1: detect violations on the raw surface
    arb_report= detect_with_forward(surface)
    n_violations_before= len(arb_report.violations)

    # step 2: build rejection set from violation offending fields
    reject_set, rejected= _build_rejection_set(arb_report.violations)

    # step 3: build cleaned surface
    cleaned_surface= _build_cleaned_surface(surface, reject_set)

    # step 4: estimate forward curve from survivors and populate per-slice r
    fwd_curve= {}
    if cleaned_surface is not None:
        fwd_curve= estimate_forward_curve(cleaned_surface)
        populate_per_slice_r(cleaned_surface, fwd_curve)

    # step 5: fit SVI (or eSSVI or SABR) on each cleaned slice
    fitted: list[FittedSlice] = []
    fitted_ssvi: list[FittedSSVISlice] = []
    fitted_sabr: list[FittedSABRSlice] = []
    repair_infeasible = False
    fallback_slices: list[float] = []
    failed_slices: list[float] = []
    sabr_mapping_failed: list[float] = []
    if cleaned_surface is not None:
        path_result: _PathFitResult = strategy.fit(cleaned_surface, fwd_curve)
        fitted = path_result.fitted
        fitted_ssvi = path_result.fitted_ssvi
        fitted_sabr = path_result.fitted_sabr
        fallback_slices = path_result.fallback_slices
        failed_slices = path_result.failed_slices
        sabr_mapping_failed = path_result.sabr_mapping_failed
        repair_infeasible = path_result.repair_infeasible

    # step 6: detect remaining violations on the fitted surface
    if fitted:
        svi_slices= [(fs.expiry_time, fs.params) for fs in fitted]
        remaining= detect_svi_surface(svi_slices)
    else:
        remaining= ArbitrageReport(violations=[])

    # A fit that leaves grid-detectable violations is not feasible, even
    # if the eSSVI H&M parameter check passed (the grid runs the raw-SVI
    # mapping at discrete k, which can surface violations the parameter
    # check misses).
    if remaining.violations:
        repair_infeasible = True

    # step 7: metrics
    metrics= RepairMetrics(
        n_rejected=len(rejected),
        n_total_quotes=n_total_quotes,
        n_slices_input=n_slices_input,
        n_slices_fitted=len(fitted),
        n_violations_before=n_violations_before,
        n_violations_after=len(remaining.violations),
    )

    # Enforce the failed_slices ordering contract: the eSSVI path
    # reassigns the list from the sequential fit result and then appends
    # the no-forward expiries, which can produce non-chronological lists
    # (a no-forward expiry sorting BEFORE a failed-fit expiry).  Every
    # path's failures are merged, sorted by expiry and deduplicated
    # before being stored in the report.
    failed_slices = sorted(set(failed_slices))

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
