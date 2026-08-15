import logging
from math import sqrt
from statistics import mean

from arbfree_vol.models.fitted import FittedSlice, FittedSSVISlice
from arbfree_vol.models.surface import VolSurface, ExpirySlice
from arbfree_vol.ssvi.model import ssvi_w, to_raw_svi_params, SSVIParams
from arbfree_vol.ssvi.term_structure import (
    fit_ssvi_surface_sequential,
    verify_hm_condition,
    verify_ssvi_calendar_free,
)
from arbfree_vol.svi.model import SVIParams

from arbfree_vol.repair.strategies._common import (
    _PathFitResult,
    _PrepStatus,
    _SlicePrep,
    _prepare_slice,
    RepairStrategy,
)

_logger = logging.getLogger(__name__)


class ESSVIStrategy:
    """Fit eSSVI slices sequentially by increasing maturity.

    Uses the Hendriks & Martini (2019) Prop 3.1 no-calendar-spread
    condition enforced as a HARD optimizer constraint:

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
    """
    name = "eSSVI"

    def fit(self, cleaned_surface: VolSurface, fwd_curve: dict[float, float]) -> _PathFitResult:
        fitted: list[FittedSlice] = []
        fitted_ssvi: list[FittedSSVISlice] = []
        fallback_slices: list[float] = []
        failed_slices: list[float] = []
        repair_infeasible = False
        sorted_slices = sorted(cleaned_surface.slices, key=lambda sl: sl.expiry_time)

        # ── eSSVI sequential path (Hendriks & Martini 2019) ────
        slices_data: list[tuple[float, list[tuple[float, float]]]] = []
        slice_meta: list[tuple[ExpirySlice, float]] = []  # (sl, F)
        no_forward_slices: list[float] = []  # expiries with no forward estimate
        for sl in sorted_slices:
            prep = _prepare_slice(sl, cleaned_surface, fwd_curve, self.name)
            if prep.status is _PrepStatus.NO_FORWARD:
                no_forward_slices.append(prep.expiry_time)
                continue
            if prep.status is _PrepStatus.TOO_FEW:
                continue
            assert prep.forward is not None
            slices_data.append((prep.expiry_time, list(prep.points)))
            slice_meta.append((sl, prep.forward))

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

        # Slices with no forward estimate were never submitted to the
        # sequential fit, so they are absent from seq_result.failed_slices.
        # Append them AFTER the reassignment above so the record survives
        # (and also covers the case where slices_data is empty).
        failed_slices.extend(no_forward_slices)

        return _PathFitResult(fitted=fitted, fitted_ssvi=fitted_ssvi, fallback_slices=fallback_slices, failed_slices=failed_slices, repair_infeasible=repair_infeasible)
