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
from arbfree_vol.ssvi.model import ssvi_w, to_raw_svi_params, SSVIParams
from arbfree_vol.ssvi.term_structure import (
    fit_ssvi_surface_sequential,
    verify_hm_condition,
    verify_ssvi_calendar_free,
    SequentialFitResult,
)
from arbfree_vol.sabr.model import sabr_total_variance, to_raw_svi_params as sabr_to_raw_svi_params
from arbfree_vol.sabr.term_structure import fit_sabr_term_structure
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
               surface: VolSurface,
               prev_slice: SVIParams | None = None) -> FittedSlice | None:

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
        params= calibrate_constrained(points, prev_slice=prev_slice)
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


def repair(surface: VolSurface, use_ssvi: bool= False, use_sabr: bool= False) -> RepairReport:
    """Repair a volatility surface by rejecting arb violating quotes,
    re-estimating the forward curve, and refitting SVI slices.

    If ``use_ssvi=True``, fits SSVI slices sequentially by increasing
    maturity with the Hendriks & Martini (2019) Prop 3.1
    no-calendar-spread condition enforced as a HARD optimizer
    constraint:

      (a) theta non-decreasing,
      (b) chi = theta*psi non-decreasing,
      (c) |(rho*chi)_{i+1} - (rho*chi)_i| / (chi_{i+1} - chi_i) <= 1.

    Both Gatheral-Jacquier (2014) butterfly bounds
    ``theta*psi*(1+|rho|) <= 4`` and ``theta*psi^2*(1+|rho|) <= 4``
    are enforced per slice.  Per-slice rho is fully free
    (tanh-reparametrised, no cross-slice functional form).  The
    discrete formulation follows Corbetta et al. (2019),
    arXiv:1804.04924, Sec 2.2-2.3.  Slices that fit within the hard
    constraints are arbitrage-free by construction; slices that fall
    back to the unconstrained fit (see ``RepairReport.fallback_slices``
    and ``repair_infeasible``) are NOT — the grid-based calendar
    detector (detect_svi_surface) is then load-bearing and reports
    those violations as remaining_violations, not merely a redundant
    regression assertion.

    If ``use_sabr=True``, fits the SABR model (Hagan et al. 2002)
    as a COMPARISON parametrisation alongside the arbitrage-certified
    eSSVI primary surface.  The SABR parameters alpha(t), nu(t) and
    rho(t) are modelled as cubic B-spline curves across expiries with
    coefficient-level reparametrisation (tanh for rho, exp+floor for
    alpha/nu) keeping curves in-range between knots via the convex-hull
    property — no runtime clamping needed.  A cross-slice calendar-arb
    SOFT penalty is included in the objective.  Verification is EMPIRICAL
    and grid-based via ``detect_svi_surface`` — NOT a closed-form /
    by-construction guarantee.  The SABR parameters are mapped to raw SVI
    via ``to_raw_svi_params``, and the native SABR parameters are stored
    in ``fitted_sabr_slices``.  Dynamic SABR is a not-implemented research
    extension.

    The SABR->SVI mapping runs under a raised evaluation budget
    (``max_nfev=50000`` in ``to_raw_svi_params``); that budget (B) reduces
    how often the per-slice mapping raises.  The mapping call is ALSO
    wrapped so that when some future real slice exceeds even the raised
    budget, ``repair()`` degrades to a logged, inspectable failure (the
    slice's expiry is recorded in ``sabr_mapping_failed_slices``) instead
    of crashing — that wrap (A) is the actual correctness guarantee, since
    no ``max_nfev`` is provably sufficient over a continuous parameter
    space.

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
    repair_infeasible= False
    fallback_slices: list[float] = []
    failed_slices: list[float] = []
    sabr_mapping_failed: list[float] = []
    if cleaned_surface is not None:
        sorted_slices = sorted(cleaned_surface.slices, key=lambda sl: sl.expiry_time)

        if use_ssvi:
            # ── eSSVI sequential path (Hendriks & Martini 2019) ────
            slices_data: list[tuple[float, list[tuple[float, float]]]] = []
            slice_meta: list[tuple[ExpirySlice, float]] = []  # (sl, F)
            for sl in sorted_slices:
                F = fwd_curve.get(sl.expiry_time)
                if F is None:
                    continue
                strike_w = slice_total_variance(cleaned_surface, sl)
                if len(strike_w) < 5:
                    _logger.warning(
                        "slice T=%.4f has %d (k,w) points — need >= 5; "
                        "skipping SSVI fit",
                        sl.expiry_time, len(strike_w),
                    )
                    continue
                pts = [
                    (log(strike / F), w)
                    for strike, w in strike_w.items()
                ]
                pts.sort()
                slices_data.append((sl.expiry_time, pts))
                slice_meta.append((sl, F))

            if slices_data:
                seq_result = fit_ssvi_surface_sequential(slices_data)
                params_list = seq_result.fitted_slices
                fallback_slices = seq_result.fallback_slices
                failed_slices = seq_result.failed_slices
                fitted_by_T: dict[float, SSVIParams] = {T: p for T, p in params_list}

                for (sl, F), (T, pts) in zip(slice_meta, slices_data):
                    ssvi_params = fitted_by_T.get(T)
                    if ssvi_params is None:
                        _logger.warning(
                            "eSSVI: no fit for T=%.4f; skipping in fitted output",
                            T,
                        )
                        continue

                    errors = [
                        (ssvi_w(k, ssvi_params.theta,
                                ssvi_params.rho, ssvi_params.psi) - w) ** 2
                        for k, w in pts
                    ]
                    rmse = sqrt(mean(errors))

                    a, b, rho_raw, m, sigma = to_raw_svi_params(
                        ssvi_params.theta, ssvi_params.rho, ssvi_params.psi
                    )
                    raw_svi_params = SVIParams(
                        a=a, b=b, rho=rho_raw, m=m, sigma=sigma
                    )

                    fitted.append(FittedSlice(
                        expiry_time=sl.expiry_time,
                        params=raw_svi_params,
                        rmse=rmse,
                        forward_price=F,
                        n_quotes_total=len(sl.quotes),
                        n_quotes_used=len(pts),
                        data_points=tuple(pts),
                    ))
                    fitted_ssvi.append(FittedSSVISlice(
                        expiry_time=sl.expiry_time,
                        ssvi=SSVIParams(
                            theta=ssvi_params.theta,
                            rho=ssvi_params.rho,
                            psi=ssvi_params.psi,
                        ),
                        rmse=rmse,
                        forward_price=F,
                        n_quotes_total=len(sl.quotes),
                        n_quotes_used=len(pts),
                        essvi=None,
                    ))

                # Check calendar-arb feasibility of the fit
                if len(params_list) >= 2:
                    params_only_sorted = [p for _, p in sorted(params_list, key=lambda x: x[0])]
                    repair_infeasible = (
                        not verify_hm_condition(params_only_sorted)
                        or not verify_ssvi_calendar_free(params_only_sorted)
                    )
                else:
                    repair_infeasible = False

        elif use_sabr:
            # ── SABR B-spline term-structure path ───────────────────
            # Build slices_data for joint fit across expiries
            sabr_slices_data: list[tuple[float, float, list[tuple[float, float]]]] = []
            sabr_meta: list[tuple[ExpirySlice, float, int]] = []  # (sl, F, n_total)

            for sl in sorted_slices:
                F = fwd_curve.get(sl.expiry_time)
                if F is None:
                    continue
                strike_w = slice_total_variance(cleaned_surface, sl)
                if len(strike_w) < 5:
                    _logger.warning(
                        "SABR term-structure: slice T=%.4f has %d (k,w) points — "
                        "need >= 5; skipping",
                        sl.expiry_time, len(strike_w),
                    )
                    continue
                pts = [
                    (log(strike / F), w)
                    for strike, w in strike_w.items()
                ]
                pts.sort()
                sabr_slices_data.append((sl.expiry_time, F, pts))
                sabr_meta.append((sl, F, len(sl.quotes)))

            if sabr_slices_data:
                try:
                    sabr_params_list = fit_sabr_term_structure(sabr_slices_data)
                except RuntimeError as exc:
                    _logger.warning(
                        "SABR term-structure fit failed: %s; marking %d slice(s) failed",
                        exc, len(sabr_meta),
                    )
                    failed_slices.extend(
                        sl.expiry_time for sl, _F, _n in sabr_meta
                    )
                else:
                    for (sl, F, n_total), sabr_params, (T_i, _F_i, pts) in zip(
                        sabr_meta, sabr_params_list, sabr_slices_data,
                    ):
                        # Per-slice RMSE in w-space
                        a = sabr_params.alpha
                        b = sabr_params.beta
                        r = sabr_params.rho
                        n = sabr_params.nu
                        errors = [
                            (sabr_total_variance(k, F, T_i, a, b, r, n) - w) ** 2
                            for k, w in pts
                        ]
                        rmse = sqrt(mean(errors))

                        # Map to raw SVI params for the SVI-based pipeline
                        try:
                            a_svi, b_svi, rho_svi, m_svi, sigma_svi = sabr_to_raw_svi_params(
                                sabr_params, F, T_i,
                            )
                        except RuntimeError as exc:
                            # The raised max_nfev budget (B) reduces how often
                            # this path is hit; this wrap (A) guarantees that
                            # when some future real slice exceeds even the
                            # raised budget, repair() degrades to a logged,
                            # inspectable failure instead of crashing.  No
                            # max_nfev is provably sufficient over a
                            # continuous parameter space.
                            _logger.warning(
                                "SABR->SVI mapping failed for slice T=%.4f "
                                "(alpha=%.6f beta=%.6f rho=%.6f nu=%.6f): %s; "
                                "slice recorded as failed-mapping",
                                sl.expiry_time, a, b, r, n, exc,
                            )
                            sabr_mapping_failed.append(sl.expiry_time)
                            continue
                        raw_svi_params = SVIParams(
                            a=a_svi, b=b_svi, rho=rho_svi, m=m_svi, sigma=sigma_svi,
                        )

                        fitted.append(FittedSlice(
                            expiry_time=sl.expiry_time,
                            params=raw_svi_params,
                            rmse=rmse,
                            forward_price=F,
                            n_quotes_total=n_total,
                            n_quotes_used=len(pts),
                            data_points=tuple(pts),
                        ))
                        fitted_sabr.append(FittedSABRSlice(
                            expiry_time=sl.expiry_time,
                            sabr=sabr_params,
                            rmse=rmse,
                            forward_price=F,
                            n_quotes_total=n_total,
                            n_quotes_used=len(pts),
                        ))
        else:
            prev_svi: SVIParams | None = None
            for sl in sorted_slices:
                F= fwd_curve.get(sl.expiry_time)
                if F is None:
                    continue
                fs= _fit_slice(sl, F, cleaned_surface, prev_slice=prev_svi)
                if fs is not None:
                    fitted.append(fs)
                    prev_svi = fs.params

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
