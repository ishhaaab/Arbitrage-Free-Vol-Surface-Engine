import logging
from math import sqrt
from statistics import mean

from arbfree_vol.models.fitted import FittedSlice, FittedSABRSlice
from arbfree_vol.models.surface import VolSurface, ExpirySlice
from arbfree_vol.sabr.model import SABRParams, sabr_total_variance, to_raw_svi_params as sabr_to_raw_svi_params
from arbfree_vol.sabr.term_structure import fit_sabr_term_structure
from arbfree_vol.svi.model import SVIParams

from arbfree_vol.repair.strategies._common import (
    _PathFitResult,
    _PrepStatus,
    _prepare_slice,
)

_logger = logging.getLogger(__name__)


def _map_sabr_to_svi(
    sabr_params: SABRParams, forward_price: float, expiry_time: float,
) -> SVIParams | None:
    """Map SABR params to raw SVI params, degrading to None instead of aborting.

    ``sabr_to_raw_svi_params`` runs under a raised evaluation budget
    (``max_nfev=50000``); that budget (B) reduces how often the per-slice
    mapping raises.  This wrap (A) is the actual correctness guarantee:
    no ``max_nfev`` is provably sufficient over a continuous parameter
    space, and scipy can raise either RuntimeError (budget exhausted) or
    ValueError (non-finite residuals), so both are caught and reported
    through the same logged, inspectable failure path.
    """
    try:
        a_svi, b_svi, rho_svi, m_svi, sigma_svi = sabr_to_raw_svi_params(
            sabr_params, forward_price, expiry_time,
        )
    except RuntimeError as exc:
        _logger.warning(
            "SABR->SVI mapping failed for slice T=%.4f "
            "(alpha=%.6f beta=%.6f rho=%.6f nu=%.6f): %s; "
            "slice recorded as failed-mapping",
            expiry_time, sabr_params.alpha, sabr_params.beta,
            sabr_params.rho, sabr_params.nu, exc,
        )
        return None
    except ValueError as exc:
        _logger.warning(
            "SABR->SVI mapping failed for slice T=%.4f "
            "(alpha=%.6f beta=%.6f rho=%.6f nu=%.6f) with "
            "ValueError: %s; slice recorded as failed-mapping",
            expiry_time, sabr_params.alpha, sabr_params.beta,
            sabr_params.rho, sabr_params.nu, exc,
        )
        return None
    return SVIParams(a=a_svi, b=b_svi, rho=rho_svi, m=m_svi, sigma=sigma_svi)


class SABRStrategy:
    """Fit the SABR model (Hagan et al. 2002) as a COMPARISON
    parametrisation alongside the arbitrage-certified eSSVI primary
    surface.

    The SABR parameters alpha(t), nu(t) and rho(t) are modelled as
    cubic B-spline curves across expiries with coefficient-level
    reparametrisation (tanh for rho, exp+floor for alpha/nu) keeping
    curves in-range between knots via the convex-hull property — no
    runtime clamping needed.  A cross-slice calendar-arb SOFT penalty
    is included in the objective.  Verification is EMPIRICAL and
    grid-based via ``detect_svi_surface`` — NOT a closed-form /
    by-construction guarantee.  The SABR parameters are mapped to raw
    SVI via ``to_raw_svi_params``, and the native SABR parameters are
    stored in ``fitted_sabr_slices``.  Dynamic SABR is a
    not-implemented research extension.

    The SABR->SVI mapping runs under a raised evaluation budget
    (``max_nfev=50000`` in ``to_raw_svi_params``); that budget (B)
    reduces how often the per-slice mapping raises.  The mapping call
    is ALSO wrapped so that when some future real slice exceeds even
    the raised budget, ``repair()`` degrades to a logged, inspectable
    failure (the slice's expiry is recorded in
    ``sabr_mapping_failed_slices``) instead of crashing — that wrap
    (A) is the actual correctness guarantee, since no ``max_nfev`` is
    provably sufficient over a continuous parameter space.
    """
    name = "SABR"

    def fit(self, cleaned_surface: VolSurface, fwd_curve: dict[float, float]) -> _PathFitResult:
        fitted: list[FittedSlice] = []
        fitted_sabr: list[FittedSABRSlice] = []
        failed_slices: list[float] = []
        sabr_mapping_failed: list[float] = []
        sorted_slices = sorted(cleaned_surface.slices, key=lambda sl: sl.expiry_time)

        # ── SABR B-spline term-structure path ───────────────────
        # Build slices_data for joint fit across expiries
        sabr_slices_data: list[tuple[float, float, list[tuple[float, float]]]] = []
        sabr_meta: list[tuple[ExpirySlice, float, int]] = []  # (sl, F, n_total)

        for sl in sorted_slices:
            prep = _prepare_slice(sl, cleaned_surface, fwd_curve, self.name)
            if prep.status is _PrepStatus.NO_FORWARD:
                failed_slices.append(prep.expiry_time)
                continue
            if prep.status is _PrepStatus.TOO_FEW:
                continue
            assert prep.forward is not None
            sabr_slices_data.append((prep.expiry_time, prep.forward, list(prep.points)))
            sabr_meta.append((sl, prep.forward, len(sl.quotes)))

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
                    raw_svi_params = _map_sabr_to_svi(sabr_params, F, T_i)
                    if raw_svi_params is None:
                        sabr_mapping_failed.append(sl.expiry_time)
                        continue

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

        return _PathFitResult(fitted=fitted, fitted_sabr=fitted_sabr, failed_slices=failed_slices, sabr_mapping_failed=sabr_mapping_failed)
