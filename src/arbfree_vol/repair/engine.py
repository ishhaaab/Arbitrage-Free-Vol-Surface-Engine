from math import log, sqrt
from statistics import mean
import logging

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType, OffendingQuote
from arbfree_vol.arbitrage.report import ArbitrageReport
from arbfree_vol.arbitrage.quote_detect import detect_with_forward
from arbfree_vol.arbitrage.svi_detect import detect_svi_surface
from arbfree_vol.svi.calibration import calibrate_constrained
from arbfree_vol.svi.model import svi_total_variance, SVIParams
from arbfree_vol.ssvi.calibration import fit_ssvi_slice
from arbfree_vol.ssvi.model import ssvi_w, to_raw_svi_params
from arbfree_vol.sabr.calibration import calibrate_sabr
from arbfree_vol.sabr.model import sabr_total_variance, to_raw_svi_params as sabr_to_raw_svi_params
from arbfree_vol.variance import slice_total_variance
from arbfree_vol.repair.report import (
    RejectedQuote,
    FittedSlice,
    FittedSSVISlice,
    FittedSABRSlice,
    RepairMetrics,
    RepairReport,
)

_logger = logging.getLogger(__name__)
from arbfree_vol.repair.fwd_curve import estimate_forward_curve, populate_per_slice_r


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


def _fit_slice(sl: ExpirySlice,
               forward_price: float,
               surface: VolSurface) -> FittedSlice | None:

    """Fit SVI to one cleaned slice using the estimated forward price.

    Returns None if fewer than 5 (k, w) points are available.
    """
    # total variance uses the surface r/q for IV solving (independent of forward)
    strike_w= slice_total_variance(surface, sl)
    if len(strike_w) < 5:
        _logger.warning(
            "slice T=%.4f has %d (k,w) points after IV solving — need >= 5; skipping",
            sl.expiry_time, len(strike_w),
        )
        return None

    # build (k, w) points using the estimated forward, not surface r/q
    points= [
        (log(strike / forward_price), w)
        for strike, w in strike_w.items()
    ]
    points.sort()

    try:
        params= calibrate_constrained(points)
    except RuntimeError:
        _logger.warning(
            "SVI constrained calibration failed for slice T=%.4f; skipping",
            sl.expiry_time, exc_info=True,
        )
        return None

    # RMSE in w-space
    errors= [
        (svi_total_variance(k, params.a, params.b, params.rho, params.m, params.sigma) - w) ** 2
        for k, w in points
    ]
    rmse= sqrt(mean(errors))

    return FittedSlice(
        expiry_time=sl.expiry_time,
        params=params,
        rmse=rmse,
        forward_price=forward_price,
        n_quotes_total=len(sl.quotes),
        n_quotes_used=len(points),
        data_points=tuple(points),
    )


def _fit_slice_ssvi(sl: ExpirySlice,
                    forward_price: float,
                    surface: VolSurface) -> tuple[FittedSlice, FittedSSVISlice] | None:
    """Fit eSSVI to one cleaned slice and map to raw SVI for visualization.

    Returns (FittedSlice, FittedSSVISlice) or None if too few points.
    """
    strike_w= slice_total_variance(surface, sl)
    if len(strike_w) < 5:
        _logger.warning(
            "slice T=%.4f has %d (k,w) points — need >= 5; skipping SSVI fit",
            sl.expiry_time, len(strike_w),
        )
        return None

    points= [
        (log(strike / forward_price), w)
        for strike, w in strike_w.items()
    ]
    points.sort()

    try:
        ssvi_params= fit_ssvi_slice(points)
    except RuntimeError:
        _logger.warning(
            "SSVI calibration failed for slice T=%.4f; skipping",
            sl.expiry_time, exc_info=True,
        )
        return None

    # RMSE in w-space using SSVI formula
    errors= [
        (ssvi_w(k, ssvi_params.theta, ssvi_params.rho, ssvi_params.psi) - w) ** 2
        for k, w in points
    ]
    rmse= sqrt(mean(errors))

    # Map to raw SVI params so existing SVI-based pipeline (plots, detection) works
    a, b, rho, m, sigma= to_raw_svi_params(
        ssvi_params.theta, ssvi_params.rho, ssvi_params.psi
    )
    raw_svi_params= SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)

    fitted_svi= FittedSlice(
        expiry_time=sl.expiry_time,
        params=raw_svi_params,
        rmse=rmse,
        forward_price=forward_price,
        n_quotes_total=len(sl.quotes),
        n_quotes_used=len(points),
        data_points=tuple(points),
    )
    fitted_ssvi= FittedSSVISlice(
        expiry_time=sl.expiry_time,
        ssvi=ssvi_params,
        rmse=rmse,
        forward_price=forward_price,
        n_quotes_total=len(sl.quotes),
        n_quotes_used=len(points),
    )
    return fitted_svi, fitted_ssvi


def _fit_slice_sabr(sl: ExpirySlice,
                    forward_price: float,
                    surface: VolSurface) -> tuple[FittedSlice, FittedSABRSlice] | None:
    """Fit SABR to one cleaned slice and map to raw SVI for visualization.

    Returns (FittedSlice, FittedSABRSlice) or None if too few points.
    """
    strike_w = slice_total_variance(surface, sl)
    if len(strike_w) < 5:
        _logger.warning(
            "slice T=%.4f has %d (k,w) points — need >= 5; skipping SABR fit",
            sl.expiry_time, len(strike_w),
        )
        return None

    points = [
        (log(strike / forward_price), w)
        for strike, w in strike_w.items()
    ]
    points.sort()

    try:
        sabr_params = calibrate_sabr(points, forward=forward_price,
                                      expiry_time=sl.expiry_time)
    except RuntimeError:
        _logger.warning(
            "SABR calibration failed for slice T=%.4f; skipping",
            sl.expiry_time, exc_info=True,
        )
        return None

    # RMSE in w-space using SABR formula
    alpha = sabr_params.alpha
    beta = sabr_params.beta
    rho = sabr_params.rho
    nu = sabr_params.nu
    errors = [
        (sabr_total_variance(k, forward_price, sl.expiry_time,
                             alpha, beta, rho, nu) - w) ** 2
        for k, w in points
    ]
    rmse = sqrt(mean(errors))

    # Map to raw SVI params so existing SVI-based pipeline works
    a, b, r, m, sigma = sabr_to_raw_svi_params(
        sabr_params, forward_price, sl.expiry_time
    )
    raw_svi_params = SVIParams(a=a, b=b, rho=r, m=m, sigma=sigma)

    fitted_svi = FittedSlice(
        expiry_time=sl.expiry_time,
        params=raw_svi_params,
        rmse=rmse,
        forward_price=forward_price,
        n_quotes_total=len(sl.quotes),
        n_quotes_used=len(points),
        data_points=tuple(points),
    )
    fitted_sabr = FittedSABRSlice(
        expiry_time=sl.expiry_time,
        sabr=sabr_params,
        rmse=rmse,
        forward_price=forward_price,
        n_quotes_total=len(sl.quotes),
        n_quotes_used=len(points),
    )
    return fitted_svi, fitted_sabr


def repair(surface: VolSurface, use_ssvi: bool= False, use_sabr: bool= False) -> RepairReport:
    """Repair a volatility surface by rejecting arb violating quotes,
    re estimating the forward curve, and refitting SVI slices.

    If ``use_ssvi=True``, fits eSSVI which is arb free by construction
    instead of raw SVI per slice.  The fitted eSSVI parameters are
    mapped back to raw SVI for the ``fitted_slices`` field, so the
    existing SVI-based visualization and detection code continues to
    work.  The native eSSVI parameters are stored in
    ``fitted_ssvi_slices``.

    If ``use_sabr=True``, fits the SABR model (Hagan et al. 2002)
    instead of raw SVI per slice.  The SABR parameters are mapped to
    raw SVI via ``to_raw_svi_params`` adapter, and the native SABR
    parameters are stored in ``fitted_sabr_slices``.

    ``use_ssvi`` and ``use_sabr`` are mutually exclusive.
    """
    if use_ssvi and use_sabr:
        raise ValueError("use_ssvi and use_sabr are mutually exclusive")

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
    fitted: list[FittedSlice]= []
    fitted_ssvi: list[FittedSSVISlice]= []
    fitted_sabr: list[FittedSABRSlice]= []
    if cleaned_surface is not None:
        for sl in cleaned_surface.slices:
            F= fwd_curve.get(sl.expiry_time)
            if F is None:
                continue
            if use_ssvi:
                result= _fit_slice_ssvi(sl, F, cleaned_surface)
                if result is not None:
                    fs, fssvi= result
                    fitted.append(fs)
                    fitted_ssvi.append(fssvi)
            elif use_sabr:
                result= _fit_slice_sabr(sl, F, cleaned_surface)
                if result is not None:
                    fs, fsabr= result
                    fitted.append(fs)
                    fitted_sabr.append(fsabr)
            else:
                fs= _fit_slice(sl, F, cleaned_surface)
                if fs is not None:
                    fitted.append(fs)

    # step 6: detect remaining violations on the fitted surface
    if fitted:
        svi_slices= [(fs.expiry_time, fs.params) for fs in fitted]
        remaining= detect_svi_surface(svi_slices)
    else:
        remaining= ArbitrageReport(violations=[])

    # step 7: metrics
    metrics= RepairMetrics(
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
        fitted_ssvi_slices=tuple(fitted_ssvi),
        fitted_sabr_slices=tuple(fitted_sabr),
        remaining_violations=remaining,
        metrics=metrics,
        cleaned_surface=cleaned_surface,
    )
