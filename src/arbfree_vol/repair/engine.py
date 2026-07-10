from math import log, sqrt
from statistics import mean

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType, OffendingQuote
from arbfree_vol.arbitrage.report import ArbitrageReport
from arbfree_vol.arbitrage.quote_detect import detect
from arbfree_vol.arbitrage.svi_detect import detect_svi_surface
from arbfree_vol.svi.calibration import calibrate
from arbfree_vol.svi.model import svi_total_variance
from arbfree_vol.variance import slice_total_variance
from arbfree_vol.repair.report import (
    RejectedQuote,
    FittedSlice,
    RepairMetrics,
    RepairReport,
)
from arbfree_vol.repair.fwd_curve import estimate_forward_curve


def _build_rejection_set(
    violations,
) -> tuple[set[tuple[float, float, OptionType]], list[RejectedQuote]]:
    """Collect all offending quotes from violations into a deduplicated set.

    Returns (identity_set, rejected_quote_list) where the set is used for
    fast lookup and the list preserves the rejection reason.
    """
    seen: set[tuple[float, float, OptionType]] = set()
    rejected: list[RejectedQuote] = []

    for v in violations:
        for oq in v.offending:
            key = (oq.strike, oq.expiry_time, oq.option_type)
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


def _build_cleaned_surface(
    surface: VolSurface,
    reject_set: set[tuple[float, float, OptionType]],
) -> VolSurface | None:
    """Remove all rejected quotes and drop empty slices."""
    cleaned: list[ExpirySlice] = []
    for sl in surface.slices:
        kept = []
        for q in sl.quotes:
            key = (q.strike, sl.expiry_time, q.option_type)
            if key not in reject_set:
                kept.append(q)
        if kept:
            cleaned.append(ExpirySlice(expiry_time=sl.expiry_time, quotes=kept))

    if not cleaned:
        return None
    return VolSurface(
        spot=surface.spot,
        risk_free=surface.risk_free,
        div_yield=surface.div_yield,
        slices=cleaned,
    )


def _fit_slice(
    sl: ExpirySlice,
    forward_price: float,
    surface: VolSurface,
) -> FittedSlice | None:
    """Fit SVI to one cleaned slice using the estimated forward price.

    Returns None if fewer than 5 (k, w) points are available.
    """
    # total variance uses the surface r/q for IV solving (independent of forward)
    strike_w = slice_total_variance(surface, sl)
    if len(strike_w) < 5:
        return None

    # build (k, w) points using the estimated forward, not surface r/q
    points = [
        (log(strike / forward_price), w)
        for strike, w in strike_w.items()
    ]
    points.sort()

    params = calibrate(points)

    # RMSE in w-space
    errors = [
        (svi_total_variance(k, params.a, params.b, params.rho, params.m, params.sigma) - w) ** 2
        for k, w in points
    ]
    rmse = sqrt(mean(errors))

    return FittedSlice(
        expiry_time=sl.expiry_time,
        params=params,
        rmse=rmse,
        forward_price=forward_price,
        n_quotes_total=len(sl.quotes),
        n_quotes_used=len(points),
    )


def repair(surface: VolSurface) -> RepairReport:
    """Repair a volatility surface by rejecting arb violating quotes,
    re estimating the forward curve, and refitting SVI slices.

    Returns a RepairReport with rejected quotes, fitted slices, remaining
    violations, quality metrics, and the cleaned surface.
    """
    n_total_quotes = sum(len(sl.quotes) for sl in surface.slices)
    n_slices_input = len(surface.slices)

    # step 1: detect violations on the raw surface
    arb_report = detect(surface)
    n_violations_before = len(arb_report.violations)

    # step 2: build rejection set from violation offending fields
    reject_set, rejected = _build_rejection_set(arb_report.violations)

    # step 3: build cleaned surface
    cleaned_surface = _build_cleaned_surface(surface, reject_set)

    # step 4: estimate forward curve from survivors
    fwd_curve = {}
    if cleaned_surface is not None:
        fwd_curve = estimate_forward_curve(cleaned_surface)

    # step 5: fit SVI on each cleaned slice
    fitted: list[FittedSlice] = []
    if cleaned_surface is not None:
        for sl in cleaned_surface.slices:
            F = fwd_curve.get(sl.expiry_time)
            if F is None:
                continue
            fs = _fit_slice(sl, F, cleaned_surface)
            if fs is not None:
                fitted.append(fs)

    # step 6: detect remaining violations on the fitted surface
    if fitted:
        svi_slices = [(fs.expiry_time, fs.params) for fs in fitted]
        remaining = detect_svi_surface(svi_slices)
    else:
        remaining = ArbitrageReport(violations=[])

    # step 7: metrics
    metrics = RepairMetrics(
        n_rejected=len(rejected),
        n_total_quotes=n_total_quotes,
        n_slices_input=n_slices_input,
        n_slices_fitted=len(fitted),
        n_violations_before=n_violations_before,
        n_violations_after=len(remaining.violations),
    )

    return RepairReport(
        rejected=tuple(rejected),
        fitted_slices=tuple(fitted),
        remaining_violations=remaining,
        metrics=metrics,
        cleaned_surface=cleaned_surface,
    )
