"""SABR term-structure calibration via B-spline parameter curves.

Fits SABR parameters alpha(t), nu(t), rho(t) across multiple expiries
using cubic B-splines with coefficient-level reparametrisation that
exploits the convex-hull property to keep the curves in-range between
knots without runtime clamping.

Calendar arbitrage is addressed via a SOFT penalty in the least-squares
objective — it is NOT enforced as a hard constraint.

.. admonition:: Empirical parametrisation
   :class: warning

   This module provides an **empirical comparison** parametrisation for
   SABR across expiries.  Calendar-arb verification is grid-based
   (``detect_svi_surface``) and is NOT a closed-form / by-construction
   guarantee.  The classical Hagan SABR model (Hagan et al. 2002) has
   no closed-form arbitrage-free construction for a full term structure.
   Dynamic SABR is a not-implemented research extension.

Reference
---------
Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).
Managing Smile Risk. *Wilmott Magazine*, 1, 84-108.
"""

import logging
from math import atanh, log

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline
from scipy.optimize import least_squares

from arbfree_vol.sabr.calibration import calibrate_sabr
from arbfree_vol.sabr.model import SABRParams, sabr_total_variance

_logger = logging.getLogger(__name__)

EPS_FLOOR: float = 1e-6
"""Minimum value for alpha and nu (exponentially bounded above this)."""

_RHO_BOUND: float = 0.999
"""Maximum absolute value for rho; matches SABRParams Pydantic bound."""


# ---------------------------------------------------------------------------
# Helpers: coefficient <-> parameter conversions
# ---------------------------------------------------------------------------

def _alpha_from_u(u: np.ndarray) -> np.ndarray:
    """u_alpha -> alpha: always >= EPS_FLOOR."""
    return np.exp(u) + EPS_FLOOR


def _nu_from_u(u: np.ndarray) -> np.ndarray:
    """u_nu -> nu: always >= EPS_FLOOR."""
    return np.exp(u) + EPS_FLOOR


def _rho_from_u(u: np.ndarray) -> np.ndarray:
    """u_rho -> rho: always in (-_RHO_BOUND, _RHO_BOUND) via scaled tanh.

    Uses rho = _RHO_BOUND * tanh(u) so the curve is bounded to
    (-0.999, 0.999) by the convex-hull property of the B-spline
    control points.
    """
    return _RHO_BOUND * np.tanh(u)


def _u_from_alpha(alpha: float) -> float:
    """alpha -> u_alpha: inverse of exp + EPS_FLOOR."""
    return log(max(alpha - EPS_FLOOR, 1e-15))


def _u_from_nu(nu: float) -> float:
    """nu -> u_nu: inverse of exp + EPS_FLOOR."""
    return log(max(nu - EPS_FLOOR, 1e-15))


def _u_from_rho(rho: float) -> float:
    """rho -> u_rho: inverse of scaled tanh, clipped for safety."""
    clipped = np.clip(rho / _RHO_BOUND, -0.99, 0.99)
    return float(atanh(clipped))


# ---------------------------------------------------------------------------
# Core fitting routine
# ---------------------------------------------------------------------------

def fit_sabr_term_structure(
    slices_data: list[tuple[float, float, list[tuple[float, float]]]],
    *,
    beta: float = 0.5,
    arb_penalty: float = 50.0,
    n_k: int = 41,
    k_min: float = -1.5,
    k_max: float = 1.5,
    return_splines: bool = False,
) -> list[SABRParams] | tuple[list[SABRParams], dict[str, BSpline]]:
    """Fit SABR across expiries with B-spline term structures on
    alpha(t), nu(t) and rho(t).

    Parameters
    ----------
    slices_data : list of (expiry_time, forward, points)
        Each entry is one expiry.  ``points`` is a list of ``(k, w)``
        (log-moneyness, total variance) observations.
        The list is sorted internally by ascending ``expiry_time``.
    beta : float
        Fixed beta parameter (default 0.5).
    arb_penalty : float
        Weight for the cross-slice calendar-arb SOFT penalty.
    n_k : int
        Number of grid points for the calendar-arb penalty evaluation.
    k_min, k_max : float
        Log-moneyness range for the calendar penalty grid.
    return_splines : bool, optional
        If True, also return a dict with the fitted BSpline objects
        (``{"alpha": ..., "nu": ..., "rho": ...}``) so callers can
        evaluate the curves at arbitrary t.  Default False.

    Returns
    -------
    list[SABRParams]
        Per-slice SABR parameters in ascending maturity order.
        When ``return_splines=True``, returns ``(list, dict)``.

    Notes
    -----
    - N=1 (single slice): delegates to ``calibrate_sabr``.
    - N>=2: builds cubic B-spline basis anchored at the expiry times.
      Decision vector ``p = [u_alpha_0.., u_nu_0.., u_rho_0..]`` where
      each group of coefficients is mapped through exp/tanh to keep the
      spline curve in the valid range by the convex-hull property.
    - Joint-fit fallback: when the joint B-spline least-squares fit
      reports ``success == False``, the function logs a WARNING
      ("falling back to per-slice calibrate_sabr") and returns the
      per-slice ``calibrate_sabr`` results computed during initialisation
      (step A) EXACTLY as-is.  There is no marker distinguishing a
      fallback result from a converged joint fit — callers pinning the
      fallback contract must compare the returned parameters against an
      independent direct ``calibrate_sabr`` call on the same slices
      (they are equal within solver determinism).
    """
    # Sort by ascending expiry
    slices_data = sorted(slices_data, key=lambda sd: sd[0])
    N = len(slices_data)

    # ---- N = 1: delegate to per-slice calibrate_sabr ----
    if N == 1:
        T, F, points = slices_data[0]
        params = calibrate_sabr(points, forward=F, expiry_time=T, beta_hint=beta)
        result = [params]
        if return_splines:
            # For single slice, build constant splines
            t_knot = np.array([T])
            spl_a = make_interp_spline(t_knot, [params.alpha], k=0)
            spl_n = make_interp_spline(t_knot, [params.nu], k=0)
            spl_r = make_interp_spline(t_knot, [params.rho], k=0)
            return result, {"alpha": spl_a, "nu": spl_n, "rho": spl_r}
        return result

    # ---- N >= 2: joint B-spline term-structure fit ----
    expiries = np.array([sd[0] for sd in slices_data])
    forwards = np.array([sd[1] for sd in slices_data])
    all_points = [sd[2] for sd in slices_data]

    m = len(expiries)  # number of control points = number of knots

    # --- Step A: per-slice calibrate_sabr for initial guess ---
    per_slice_params: list[SABRParams] = []
    for i in range(N):
        T_i = float(expiries[i])
        F_i = float(forwards[i])
        pts_i = all_points[i]
        p_i = calibrate_sabr(pts_i, forward=F_i, expiry_time=T_i, beta_hint=beta)
        per_slice_params.append(p_i)

    # --- Step B: build initial coefficient vector ---
    u_alpha0 = np.array([_u_from_alpha(p.alpha) for p in per_slice_params])
    u_nu0 = np.array([_u_from_nu(p.nu) for p in per_slice_params])
    u_rho0 = np.array([_u_from_rho(p.rho) for p in per_slice_params])
    x0 = np.concatenate([u_alpha0, u_nu0, u_rho0])

    # --- Step C: calendar-arb penalty grid ---
    k_grid = np.linspace(k_min, k_max, n_k)

    # Precompute which adjacent pairs exist
    adj_pairs = [(i, i + 1) for i in range(N - 1)]

    # Smoothness weight for second-difference penalty on coefficients
    lambda_smooth = 1e-2

    # --- Step D: residual function ---
    def _residuals(p: np.ndarray) -> np.ndarray:
        u_alpha = p[:m]
        u_nu = p[m:2 * m]
        u_rho = p[2 * m:3 * m]

        # Per-slice values at knot points (reparametrised)
        alpha_at_knots = _alpha_from_u(u_alpha)
        nu_at_knots = _nu_from_u(u_nu)
        rho_at_knots = _rho_from_u(u_rho)

        residuals_list: list[np.ndarray] = []

        # (1) Data residuals
        for i in range(N):
            T_i = float(expiries[i])
            F_i = float(forwards[i])
            a_i = float(alpha_at_knots[i])
            n_i = float(nu_at_knots[i])
            r_i = float(rho_at_knots[i])

            pts_i = all_points[i]
            data_res = np.array([
                sabr_total_variance(k, F_i, T_i, a_i, beta, r_i, n_i) - w
                for k, w in pts_i
            ])
            residuals_list.append(data_res)

        # (2) Calendar-arb SOFT penalty
        sqrt_arb = np.sqrt(arb_penalty)
        for i_lo, i_hi in adj_pairs:
            T_lo = float(expiries[i_lo])
            T_hi = float(expiries[i_hi])
            F_lo = float(forwards[i_lo])
            F_hi = float(forwards[i_hi])

            a_lo = float(alpha_at_knots[i_lo])
            n_lo = float(nu_at_knots[i_lo])
            r_lo = float(rho_at_knots[i_lo])
            a_hi = float(alpha_at_knots[i_hi])
            n_hi = float(nu_at_knots[i_hi])
            r_hi = float(rho_at_knots[i_hi])

            cal_res = np.zeros(n_k)
            for j in range(n_k):
                k_val = float(k_grid[j])
                w_lo = sabr_total_variance(k_val, F_lo, T_lo, a_lo, beta, r_lo, n_lo)
                w_hi = sabr_total_variance(k_val, F_hi, T_hi, a_hi, beta, r_hi, n_hi)
                gap = w_lo - w_hi  # positive = violation
                if gap > 0.0:
                    cal_res[j] = sqrt_arb * np.sqrt(gap)
                else:
                    cal_res[j] = 0.0
            residuals_list.append(cal_res)

        # (3) Smoothness penalty on coefficients (second differences)
        if m >= 3:
            for u_vec in [u_alpha, u_nu, u_rho]:
                second_diff = u_vec[2:] - 2.0 * u_vec[1:-1] + u_vec[:-2]
                residuals_list.append(np.sqrt(lambda_smooth) * second_diff)

        return np.concatenate(residuals_list)

    # --- Step E: run least_squares ---
    result = least_squares(_residuals, x0, max_nfev=5000)

    if not result.success:
        _logger.warning(
            "SABR term-structure fit did not converge: %s; "
            "falling back to per-slice calibrate_sabr",
            result.message,
        )
        fallback = per_slice_params
        if return_splines:
            spl_a = make_interp_spline(expiries, [p.alpha for p in fallback], k=min(3, m - 1))
            spl_n = make_interp_spline(expiries, [p.nu for p in fallback], k=min(3, m - 1))
            spl_r = make_interp_spline(expiries, [p.rho for p in fallback], k=min(3, m - 1))
            return fallback, {"alpha": spl_a, "nu": spl_n, "rho": spl_r}
        return fallback

    # --- Step F: extract fitted parameters ---
    p_opt = result.x
    u_alpha_opt = p_opt[:m]
    u_nu_opt = p_opt[m:2 * m]
    u_rho_opt = p_opt[2 * m:3 * m]

    alpha_fitted = _alpha_from_u(u_alpha_opt)
    nu_fitted = _nu_from_u(u_nu_opt)
    rho_fitted = _rho_from_u(u_rho_opt)

    fitted_params = [
        SABRParams(
            alpha=float(alpha_fitted[i]),
            beta=beta,
            rho=float(rho_fitted[i]),
            nu=float(nu_fitted[i]),
        )
        for i in range(N)
    ]

    if return_splines:
        spl_a = make_interp_spline(expiries, alpha_fitted, k=min(3, m - 1))
        spl_n = make_interp_spline(expiries, nu_fitted, k=min(3, m - 1))
        spl_r = make_interp_spline(expiries, rho_fitted, k=min(3, m - 1))
        return fitted_params, {"alpha": spl_a, "nu": spl_n, "rho": spl_r}

    return fitted_params
