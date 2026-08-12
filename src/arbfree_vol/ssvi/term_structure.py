"""Calendar-arbitrage-free eSSVI term-structure calibration.

Fits SSVI slices sequentially by increasing maturity, enforcing the
Hendriks & Martini (2019) Prop 3.1 no-calendar-spread condition as
HARD optimizer constraints.  Per-slice rho stays fully free
(tanh-reparametrised, no cross-slice functional form).

The discrete sequential calibration procedure follows
Corbetta, Cohort, Laachir & Martini (2019), Sec 2.2-2.3
(arXiv:1804.04924).

Conditions enforced
-------------------
For slices ordered by increasing maturity T_1 < T_2 < ... < T_N,
let  p_i  = SSVIParams.psi  (the angle / wing function),
     theta_i = SSVIParams.theta,
     rho_i   = SSVIParams.rho,
     chi_i   = theta_i * p_i.

The surface is free of calendar-spread arbitrage iff (Prop 3.1):

  (a) theta_1 <= theta_2 <= ... <= theta_N           (non-decreasing ATM variance)
  (b) chi_1   <= chi_2   <= ... <= chi_N             (non-decreasing wing magnitude)
  (c) for each adjacent pair (i, i+1):
      | (rho_{i+1}*chi_{i+1} - rho_i*chi_i) / (chi_{i+1} - chi_i) | <= 1

Butterfly arbitrage per slice (Gatheral & Jacquier 2014, Theorem 4.2):
  theta * psi * (1 + |rho|) < 4   (STRICT; enforced with a small margin)
  theta * psi^2 * (1 + |rho|) <= 4   (non-strict)
Both are written as smooth pairs of inequalities using (1+rho) and (1-rho).

References
----------
.. [HM19] Hendriks, S. & Martini, C. (2019). "The Extended SSVI
   Volatility Surface", J. Comput. Finance 22(5), 25-39.  Prop 3.1.
.. [CCLM19] Corbetta, J., Cohort, P., Laachir, I. & Martini, C.
   (2019). "Robust calibration and arbitrage-free interpolation of
   SSVI slices", arXiv:1804.04924, Sec 2.2-2.3.
.. [GJ14] Gatheral, J. & Jacquier, A. (2014). "Arbitrage-free SVI
   volatility surfaces", Quant. Finance 14(1), 59-71.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import sqrt
from statistics import mean

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, NonlinearConstraint, Bounds

from arbfree_vol.ssvi.model import SSVIParams, ssvi_w, _GJ_STRICT_EPS
from arbfree_vol.ssvi.calibration import fit_ssvi_slice

_logger = logging.getLogger(__name__)

# Margin applied to the two STRICT Gatheral-Jacquier condition-1
# residuals.  GJ (2014) Theorem 4.2 makes condition 1 STRICT
# (``theta*psi*(1+|rho|) < 4``) while condition 2 is non-strict
# (``theta*psi^2*(1+|rho|) <= 4``).  scipy optimizer constraints are
# closed sets — a bare ``> 0`` cannot be expressed, so an exact equality
# with the condition-1 boundary would be accepted as feasible.  We
# approximate the paper's strictness by requiring a small positive
# margin (>= this eps) on the two condition-1 residuals in
# ``_butterfly_constraints``.
#
# This is an ALIAS of the canonical ``_GJ_STRICT_EPS`` defined once in
# ``ssvi/model.py`` — the production constraint path and the public
# strict-mode diagnostic ``gatheral_jacquier_condition(strict=True)``
# share one constant and can never diverge.
_GJ_CONDITION1_STRICT_EPS: float = _GJ_STRICT_EPS

# Floors applied to the two Hendriks-Martini calendar constraints in
# ``_fit_slice`` (non-decreasing theta, non-decreasing chi).  Hoisted to
# named module constants so the post-fit H&M margin check below can
# reference the same eps values it must sit 10x away from.
_EPS_THETA: float = 1e-9
_EPS_CHI: float = 1e-6

# ── Post-fit margin check for degenerate H&M boundary corners (m66) ──
# docs/code_review_findings.md §6.7: a hard eSSVI fit can converge to a
# feasible-but-wrong corner pinned exactly ON the H&M Prop 3.1 boundary
# (theta_delta = eps_theta, chi_delta = eps_chi, ratio ~ 1.0) with an
# anomalously bad per-slice RMSE, and the optimizer reports it as a
# successful certified arb-free fit.  Measured m66 corner (mutmut_66 /
# the dip fixture): theta_delta = 9.99e-10, chi_delta = 1.0000e-6,
# ratio = 0.9998, hard RMSE = 0.0499 vs unconstrained RMSE = 1.6e-11.
# These thresholds are provisional and pending review — see
# docs/code_review_findings.md §6.7 "Proposed fix direction".
_HM_BOUNDARY_MARGIN_THETA: float = 1e-8   # 10x eps_theta
_HM_BOUNDARY_MARGIN_CHI: float = 1e-5     # 10x eps_chi
_HM_BOUNDARY_MARGIN_RATIO: float = 1e-3
_HM_RMSE_RATIO_MAX: float = 5.0
_HM_RMSE_FLOOR: float = 1e-9


@dataclass
class SequentialFitResult:
    """Result of a sequential eSSVI term-structure fit.

    Attributes
    ----------
    fitted_slices : list of (expiry_time, SSVIParams)
        Successful fits — either hard-constrained (H&M Prop 3.1) or
        unconstrained fallback.  Slices where *both* fits failed are
        absent.
    fallback_slices : list of float
        Expiry times where the hard-constrained fit failed but the
        unconstrained per-slice fallback succeeded.  These slices are
        NOT arbitrage-free by construction.
    failed_slices : list of float
        Expiry times where both the hard-constrained and the
        unconstrained fit failed.  These slices are omitted from
        ``fitted_slices`` entirely.
    fitted_slices_prev : list of float | None
        For each entry in ``fitted_slices``, the expiry time of the
        actual ``prev`` slice used in the calibration.  The first
        fitted slice has ``None`` (no predecessor).  For a fallback
        slice, this records the T of the last *hard-constrained* fit
        before it — which may be several positions back in the
        fitted sequence when there are consecutive fallbacks.
    """
    fitted_slices: list[tuple[float, SSVIParams]]
    fallback_slices: list[float]
    failed_slices: list[float]
    fitted_slices_prev: list[float | None] = field(default_factory=list)


def _butterfly_constraints(
    theta: float, rho: float, p: float,
) -> NDArray[np.float64]:
    """Return the four Gatheral-Jacquier butterfly residual values.

    Each residual is  ``4 - lhs >= 0``  for a safe slice.  The four
    values correspond to the smooth split of the two GJ bounds into
    pairs using (1+rho) and (1-rho) instead of (1+|rho|):

    .. math::
        4 - \\theta\\,p\\,(1+\\rho) \\ge 0, \\quad
        4 - \\theta\\,p\\,(1-\\rho) \\ge 0, \\\\
        4 - \\theta\\,p^2\\,(1+\\rho) \\ge 0, \\quad
        4 - \\theta\\,p^2\\,(1-\\rho) \\ge 0.

    The first two residuals (linear in ``p``) are the smooth split of
    Gatheral-Jacquier condition 1, ``theta*p*(1+|rho|) < 4``, which is
    STRICT in Theorem 4.2; the last two (quadratic in ``p``) are the
    split of condition 2, ``theta*p^2*(1+|rho|) <= 4``, which is
    non-strict.  Because scipy constraints are closed sets (a bare
    ``> 0`` cannot be expressed), the two condition-1 residuals are
    shifted by ``_GJ_CONDITION1_STRICT_EPS`` so that an exact equality
    with the condition-1 boundary is rejected as infeasible.  The two
    condition-2 residuals are left unshifted — the boundary is allowed.

    Reference: Gatheral & Jacquier (2014), Theorem 4.2.
    """
    return np.array([
        4.0 - theta * p * (1.0 + rho) - _GJ_CONDITION1_STRICT_EPS,
        4.0 - theta * p * (1.0 - rho) - _GJ_CONDITION1_STRICT_EPS,
        4.0 - theta * p * p * (1.0 + rho),
        4.0 - theta * p * p * (1.0 - rho),
    ])


def _fit_slice(
    points: list[tuple[float, float]],
    prev: SSVIParams | None = None,
    *,
    eps_theta: float = _EPS_THETA,
    eps_chi: float = _EPS_CHI,
) -> SSVIParams:
    """Fit a single SSVI slice subject to hard no-arbitrage constraints.

    Parameters
    ----------
    points : list of (k, w)
        Log-moneyness / total-variance pairs for this slice.
    prev : SSVIParams or None
        Previous (shorter-maturity) slice parameters.  When given the
        Hendriks-Martini calendar constraints are added.
    eps_theta, eps_chi : float
        Small positive floors for the theta and chi monotonicity
        constraints.

    Returns
    -------
    SSVIParams

    Raises
    ------
    RuntimeError
        If the optimizer cannot satisfy all hard constraints.

    Reference: Hendriks & Martini (2019) Prop 3.1; Corbetta et al.
    (2019) Sec 2.2-2.3.
    """
    if len(points) < 5:
        raise ValueError("Need at least 5 points to fit SSVI slice")

    ks = np.array([k for k, _ in points], dtype=np.float64)
    ws = np.array([w for _, w in points], dtype=np.float64)

    # ── Seed from unconstrained least-squares ──────────────────────
    from scipy.optimize import least_squares as _ls

    def _seed_resid(p):
        th, rh, ps = p
        return np.array([
            ssvi_w(float(k), th, rh, ps) - float(w)
            for k, w in points
        ])

    seed_result = _ls(
        _seed_resid,
        x0=[float(np.min(ws)), 0.0, 0.5],
        bounds=([1e-6, -0.999, 1e-6], [10.0, 0.999, 20.0]),
    )
    theta0, rho0, p0 = [float(v) for v in seed_result.x]

    # Adjust seed for calendar constraints if a predecessor is given.
    if prev is not None:
        prev_chi = prev.theta * prev.psi
        theta0 = max(prev.theta + eps_theta, theta0)
        p0 = max(p0, 1e-6)
        chi0 = theta0 * p0
        if chi0 < prev_chi + eps_chi:
            p0 = (prev_chi + eps_chi) / theta0

    rho0 = float(np.clip(rho0, -0.99, 0.99))
    p0 = float(np.clip(p0, 1e-6, 20.0))

    u0 = float(np.arctanh(rho0))
    v0 = float(np.log(p0))
    x0 = np.array([theta0, u0, v0], dtype=np.float64)

    # ── Variable bounds ────────────────────────────────────────────
    bounds = Bounds(
        lb=[1e-6, -6.0, float(np.log(1e-8))],
        ub=[10.0,  6.0, float(np.log(20.0))],
    )

    # ── Objective: sum of squared residuals ────────────────────────
    def _objective(x: NDArray[np.float64]) -> float:
        theta, u, v = x
        rho = float(np.tanh(u))
        p = float(np.exp(v))
        return float(np.sum(
            (np.array([ssvi_w(float(k), theta, rho, p) for k in ks]) - ws) ** 2
        ))

    # ── Hard constraints ───────────────────────────────────────────
    constraints: list = []

    # Butterfly constraints: four >= 0 residuals per slice
    def _bf_con(x: NDArray[np.float64]) -> NDArray[np.float64]:
        theta, u, v = x
        rho = float(np.tanh(u))
        p = float(np.exp(v))
        return _butterfly_constraints(theta, rho, p)

    constraints.append(NonlinearConstraint(_bf_con, 0.0, np.inf))

    # Calendar constraints when a predecessor exists
    if prev is not None:
        prev_chi = prev.theta * prev.psi

        # (a) theta non-decreasing
        def _theta_nd(x: NDArray[np.float64]) -> float:
            return x[0] - prev.theta

        constraints.append(NonlinearConstraint(_theta_nd, eps_theta, np.inf))

        # (b) chi non-decreasing
        def _chi_nd(x: NDArray[np.float64]) -> float:
            theta, u, v = x
            return theta * float(np.exp(v)) - prev_chi

        constraints.append(NonlinearConstraint(_chi_nd, eps_chi, np.inf))

        # (c) | rho_{i+1}*chi_{i+1} - rho_i*chi_i | / (chi_{i+1}-chi_i) <= 1
        #     written as two linear-fractional inequalities
        rho_prev_chi_prev = prev.rho * prev_chi

        def _ratio_upper(x: NDArray[np.float64]) -> float:
            theta, u, v = x
            rho = float(np.tanh(u))
            chi = theta * float(np.exp(v))
            denom = max(chi - prev_chi, eps_chi)
            return (rho * chi - rho_prev_chi_prev) / denom

        def _ratio_lower(x: NDArray[np.float64]) -> float:
            theta, u, v = x
            rho = float(np.tanh(u))
            chi = theta * float(np.exp(v))
            denom = max(chi - prev_chi, eps_chi)
            return -(rho * chi - rho_prev_chi_prev) / denom

        constraints.append(NonlinearConstraint(_ratio_upper, -1.0, 1.0))
        constraints.append(NonlinearConstraint(_ratio_lower, -1.0, 1.0))

    # ── Optimise ───────────────────────────────────────────────────
    def _run(method: str, x_init, tol: float, maxiter: int):
        opts: dict = {"maxiter": maxiter}
        if method == "trust-constr":
            opts["gtol"] = tol
        else:  # SLSQP
            opts["ftol"] = tol
        return minimize(
            _objective,
            x_init,
            method=method,
            bounds=bounds,
            constraints=constraints,
            options=opts,
        )

    # Primary attempt: trust-constr
    result = _run("trust-constr", x0, tol=1e-10, maxiter=500)
    # trust-constr: result.success is True exactly for statuses 1/2
    # (gtol/xtol satisfied).  Status 0 (max f-evals) and status 4
    # ("minimize successful but constraints not satisfied") are
    # failures, and status 3 (callback termination) needs a callback
    # this code never passes.  Only result.success is trustable.
    success = result.success

    # Retry with SLSQP if the primary run did not converge
    if not success:
        _logger.debug(
            "trust-constr did not converge (status=%s, msg=%s); "
            "retrying with SLSQP",
            getattr(result, "status", "?"), result.message,
        )
        result = _run("SLSQP", result.x, tol=1e-12, maxiter=1000)
        # SLSQP: result.success is True ONLY for exit mode 0
        # ("Optimization terminated successfully").  Modes 1 (stalled
        # line search), 2 (degenerate problem) and 3 (LSQ-subproblem
        # iteration cap) are NOT convergence — accepting them would
        # certify a non-converged fit as hard-constrained arb-free and
        # skip the fallback bookkeeping.  Anything else must raise so
        # the caller routes the slice into fallback_slices.
        success = result.success

    if not success:
        raise RuntimeError(
            f"eSSVI slice fit failed after retry: {result.message}"
        )

    theta, u, v = result.x
    return SSVIParams(
        theta=float(theta),
        rho=float(np.tanh(u)),
        psi=float(np.exp(v)),
    )


def _slice_rmse(
    params: SSVIParams, points: list[tuple[float, float]],
) -> float:
    """Root-mean-square total-variance error of ``params`` over ``points``.

    ``sqrt(mean((ssvi_w(k, theta, rho, psi) - w)^2))`` over every
    ``(k, w)`` in ``points``.  Used by the post-fit H&M margin check to
    compare the hard-constrained fit's per-slice residual with the
    unconstrained fit's baseline.
    """
    return sqrt(mean(
        (ssvi_w(k, params.theta, params.rho, params.psi) - w) ** 2
        for k, w in points
    ))


def _hm_boundary_deltas(
    prev: SSVIParams, params: SSVIParams,
) -> tuple[float, float, float]:
    """Compute the H&M Prop 3.1 boundary deltas for a candidate hard fit.

    Returns ``(theta_delta, chi_delta, ratio)`` where
    ``theta_delta = theta - theta_prev``,
    ``chi_delta = chi - chi_prev`` (``chi = theta * psi``) and
    ``ratio = |rho*chi - rho_prev*chi_prev| / max(chi_delta, eps_chi)``.

    Both the degenerate-corner predicate (:func:`_hard_fit_is_degenerate_corner`)
    and its logging call site in ``fit_ssvi_surface_sequential`` must use
    this single source of truth, so the decision and the log can never
    disagree.
    """
    theta_delta = params.theta - prev.theta
    chi_delta = params.theta * params.psi - prev.theta * prev.psi
    ratio = abs(
        params.rho * params.theta * params.psi
        - prev.rho * prev.theta * prev.psi
    ) / max(chi_delta, _EPS_CHI)
    return theta_delta, chi_delta, ratio


def _within_boundary_window(theta_delta: float, chi_delta: float,
                            ratio: float) -> bool:
    """True iff the hard fit sits within the H&M boundary margin window."""
    return (
        theta_delta <= _HM_BOUNDARY_MARGIN_THETA
        and chi_delta <= _HM_BOUNDARY_MARGIN_CHI
        and ratio >= 1.0 - _HM_BOUNDARY_MARGIN_RATIO
    )


def _hard_fit_is_degenerate_corner(
    prev: SSVIParams | None,
    params: SSVIParams,
    points: list[tuple[float, float]],
) -> bool:
    """Detect a hard eSSVI fit pinned on the H&M Prop 3.1 boundary.

    A fit is flagged through either of two paths:

    **Path A — baseline available: two signals must AGREE.**

    1. **Boundary proximity** — the hard fit landed within a small
       margin of the H&M calendar-arb boundary that ``_fit_slice``
       enforces with its eps floors:

       - ``theta_delta <= _HM_BOUNDARY_MARGIN_THETA`` (10x eps_theta)
       - ``chi_delta   <= _HM_BOUNDARY_MARGIN_CHI`` (10x eps_chi)
       - ``ratio >= 1 - _HM_BOUNDARY_MARGIN_RATIO``

       where ``theta_delta``/``chi_delta`` are the deltas vs ``prev`` and
       ``ratio`` is ``|rho*chi - rho_prev*chi_prev| / chi_delta``.

    2. **Bad per-slice RMSE** — the hard fit's RMSE over ``points``
       exceeds ``max(_HM_RMSE_RATIO_MAX * unconstrained_rmse,
       _HM_RMSE_FLOOR)``.  The unconstrained per-slice fit
       (:func:`fit_ssvi_slice`) is the RMSE baseline: a
       boundary-adjacent fit that genuinely matches the data (e.g. a
       legitimate near-flat chi pair) has a small RMSE and is NOT
       flagged.

    **Path B — baseline unavailable: boundary proximity alone flags.**
    When the unconstrained baseline fit raises, the RMSE comparison
    cannot be computed — but a fit pinned within the boundary window is
    exactly the knife-edge pattern this check exists to catch, so it IS
    flagged and routed to the fallback rather than silently certified.
    Fits outside the boundary window are never flagged, baseline or no
    baseline.

    A fit flagged here is not a genuine arb-free solution — it is an
    optimizer knife-edge that converged to a feasible-but-wrong corner.
    This is the m66 / mutmut_66 pattern (docs/code_review_findings.md
    §6.7): measured corner theta_delta = 9.99e-10, chi_delta = 1.0000e-6,
    ratio = 0.9998, hard RMSE = 0.0499 vs unconstrained RMSE = 1.6e-11.
    ``fit_ssvi_surface_sequential`` routes flagged fits to the
    unconstrained fallback.

    The first slice has no H&M predecessor boundary to sit on, so
    ``prev=None`` is never flagged.

    Returns
    -------
    bool
        ``True`` iff the fit is within the boundary margin AND (its RMSE
        is anomalously bad relative to the unconstrained fit, OR the
        unconstrained baseline fit is unavailable).
    """
    if prev is None:
        return False

    theta_delta, chi_delta, ratio = _hm_boundary_deltas(prev, params)

    if not _within_boundary_window(theta_delta, chi_delta, ratio):
        # Fast path: far from the boundary, no need for the unconstrained
        # baseline fit.
        return False

    hard_rmse = _slice_rmse(params, points)
    try:
        unconstrained = fit_ssvi_slice(points)
    except (RuntimeError, ValueError):
        # No baseline to compare against — but a fit pinned on the H&M
        # boundary is precisely the m66 pattern that must never be
        # silently certified, so flag it and let the caller route it to
        # the honest fallback.
        return True
    unconstrained_rmse = _slice_rmse(unconstrained, points)
    bad_rmse = hard_rmse > max(
        _HM_RMSE_RATIO_MAX * unconstrained_rmse, _HM_RMSE_FLOOR
    )
    return bad_rmse


def fit_ssvi_surface_sequential(
    slices_data: list[tuple[float, list[tuple[float, float]]]],
) -> SequentialFitResult:
    """Fit a sequence of SSVI slices with calendar-arb-free constraints.

    Slices are sorted by ascending maturity and fitted one at a time.
    Each slice inherits the Hendriks-Martini Prop 3.1 constraints from
    its predecessor.  Per-slice rho is fully free (tanh-reparametrised,
    no cross-slice functional form).

    When the hard-constrained fit fails for a slice (the optimizer
    cannot satisfy the H&M Prop 3.1 constraints given the data), the
    function falls back to the unconstrained per-slice
    :func:`fit_ssvi_slice`.  The fallback slice is NOT arb-free by
    construction, but the engine reports this honestly via
    ``verify_hm_condition`` and the ``repair_infeasible`` flag.  The
    fallback result is NOT used as ``prev`` for the next slice's
    constraints — ``prev`` remains the last *hard-constrained*
    successful fit.

    A hard-constrained fit that the optimizer reports as successful is
    additionally checked by :func:`_hard_fit_is_degenerate_corner`: a
    fit pinned within eps of the H&M Prop 3.1 boundary whose per-slice
    RMSE is anomalously bad relative to the unconstrained fit (the m66
    degenerate-corner pattern, docs/code_review_findings.md §6.7) is
    routed through the same ``RuntimeError`` fallback path.  When the
    unconstrained baseline fit is unavailable, the boundary proximity
    alone is enough to flag the corner — either way, a feasible-but-wrong
    boundary corner is never silently certified arb-free.

    Parameters
    ----------
    slices_data : list of (expiry_time, points)
        Each element is ``(T, [(k, w), ...])``.  Slices are sorted
        internally by ascending ``T``.

    Returns
    -------
    SequentialFitResult
        Contains ``fitted_slices`` (successful fits, hard-constrained
        or fallback), ``fallback_slices`` (T values where only the
        unconstrained fallback succeeded), and ``failed_slices``
        (T values where both fits failed).

    Reference
    ----------
    Hendriks & Martini (2019) Prop 3.1; Corbetta et al. (2019) Sec 2.2-2.3.
    """
    ordered = sorted(slices_data, key=lambda sd: sd[0])

    fitted: list[tuple[float, SSVIParams]] = []
    fallback: list[float] = []
    failed: list[float] = []
    fitted_slices_prev: list[float | None] = []
    last_valid_prev: SSVIParams | None = None
    last_valid_prev_T: float | None = None

    for expiry, pts in ordered:
        # Record the prev_T that will be used for this slice
        prev_T_for_this_slice = last_valid_prev_T

        try:
            params = _fit_slice(pts, prev=last_valid_prev)
            if last_valid_prev is not None and _hard_fit_is_degenerate_corner(last_valid_prev, params, pts):
                _theta_delta, _chi_delta, _ratio = _hm_boundary_deltas(last_valid_prev, params)
                _logger.warning("eSSVI hard fit for T=%.4f is a degenerate H&M boundary corner (theta_delta=%.3e, chi_delta=%.3e, ratio=%.6f, hard_rmse=%.4e); routing to fallback", expiry, _theta_delta, _chi_delta, _ratio, _slice_rmse(params, pts))
                raise RuntimeError("hard fit is a degenerate H&M boundary corner")
            last_valid_prev = params  # update only on hard-constrained success
            last_valid_prev_T = expiry  # update for NEXT slice
            fitted.append((expiry, params))
            fitted_slices_prev.append(prev_T_for_this_slice)
        except RuntimeError as e:
            _logger.warning(
                "eSSVI hard-constrained fit failed for T=%.4f (%s); "
                "falling back to unconstrained per-slice fit",
                expiry, e,
            )
            try:
                params = fit_ssvi_slice(pts)
                fitted.append((expiry, params))
                fallback.append(expiry)
                fitted_slices_prev.append(prev_T_for_this_slice)
                # do NOT update last_valid_prev or last_valid_prev_T
                # — fallback slices aren't arb-free
            except (RuntimeError, ValueError) as e2:
                _logger.error(
                    "eSSVI fallback fit also failed for T=%.4f (%s); "
                    "skipping this slice",
                    expiry, e2,
                )
                failed.append(expiry)
                # failed slices are NOT in fitted_slices, so no entry
                # in fitted_slices_prev

    if len(fitted) >= 2:
        params_only = [p for _, p in fitted]
        if not verify_hm_condition(params_only):
            _logger.warning(
                "verify_hm_condition reports violation after fit; "
                "likely one or more slices fell back to unconstrained. "
                "These are reported as remaining violations."
            )

    return SequentialFitResult(
        fitted_slices=fitted,
        fallback_slices=fallback,
        failed_slices=failed,
        fitted_slices_prev=fitted_slices_prev,
    )


def verify_hm_condition(
    params_seq: list[SSVIParams],
    *,
    tol: float = 1e-8,
) -> bool:
    """Check the Hendriks-Martini Prop 3.1 no-calendar-spread conditions.

    Parameters
    ----------
    params_seq : list of SSVIParams
        Ordered by ascending maturity.
    tol : float
        Numerical tolerance on the inequality checks.

    Returns
    -------
    bool
        ``True`` iff all three conditions hold within tolerance.

    Conditions checked:

    (a) theta non-decreasing  ``theta_i <= theta_{i+1} + tol``
    (b) chi = theta*psi non-decreasing  ``chi_i <= chi_{i+1} + tol``
    (c) For each adjacent pair:
        ``| rho_{i+1}*chi_{i+1} - rho_i*chi_i | / max(chi_{i+1}-chi_i, tol)
        <= 1 + tol``

    Reference: Hendriks & Martini (2019), J. Comput. Finance 22(5),
    Prop 3.1.
    """
    n = len(params_seq)
    if n <= 1:
        return True

    chis = [p.theta * p.psi for p in params_seq]

    for i in range(n - 1):
        # (a) theta non-decreasing
        if params_seq[i + 1].theta < params_seq[i].theta - tol:
            return False

        # (b) chi non-decreasing
        if chis[i + 1] < chis[i] - tol:
            return False

        # (c) | rho_{i+1}*chi_{i+1} - rho_i*chi_i | / (chi_{i+1}-chi_i) <= 1
        denom = chis[i + 1] - chis[i]
        denom = max(denom, tol)
        numerator = (
            params_seq[i + 1].rho * chis[i + 1]
            - params_seq[i].rho * chis[i]
        )
        if abs(numerator) / denom > 1.0 + tol:
            return False

    return True


def verify_ssvi_calendar_free(
    params_seq: list[SSVIParams],
    *,
    k_grid: NDArray[np.float64] | None = None,
    tol: float = 1e-4,
) -> bool:
    """Post-fit calendar-arbitrage verification on native eSSVI slices.

    ``verify_hm_condition`` checks the Hendriks & Martini Prop 3.1
    parameter conditions, which are necessary but not sufficient for
    calendar-spread absence: a pair can satisfy theta/chi monotonicity
    and the ``|ratio| <= 1`` bound yet still cross in the wings
    (``w_{i+1}(k) < w_i(k)`` for some ``k``).  This function checks the
    actual no-calendar-spread condition directly on the native SSVI
    slices over a dense log-moneyness grid.

    Parameters
    ----------
    params_seq : list of SSVIParams
        Ordered by ascending maturity.
    k_grid : NDArray[np.float64], optional
        Log-moneyness grid.  Defaults to ``linspace(-3, 3, 241)`` — the
        same range the SABR-to-SVI mapping uses.
    tol : float
        Absolute tolerance on the total-variance gap (the codebase's de
        facto arb tolerance of ``1e-4``).

    Returns
    -------
    bool
        ``True`` iff for every adjacent pair and every grid point
        ``w_{i+1}(k) >= w_i(k) - tol``.

    This is a discrete check: violations strictly between grid points or
    beyond the grid are not certified.  It complements, not replaces,
    ``verify_hm_condition``.
    """
    if params_seq is None or len(params_seq) < 2:
        return True
    if k_grid is None:
        k_grid = np.linspace(-3.0, 3.0, 241)
    for i in range(len(params_seq) - 1):
        t1, r1, p1 = params_seq[i].theta, params_seq[i].rho, params_seq[i].psi
        t2, r2, p2 = params_seq[i + 1].theta, params_seq[i + 1].rho, params_seq[i + 1].psi
        for k in k_grid:
            if ssvi_w(float(k), t1, r1, p1) - ssvi_w(float(k), t2, r2, p2) > tol:
                return False
    return True


def verify_hm_condition_breakdown(
    fitted_slices: list[tuple[float, SSVIParams]],
    fitted_prev_Ts: list[float | None] | None = None,
    *,
    tol: float = 1e-8,
) -> list[dict]:
    """Return per-fitted-slice H&M Prop 3.1 sub-condition breakdown.

    For each fitted slice (except the first if no valid predecessor),
    reports which of the three H&M sub-conditions fails relative to the
    slice's actual predecessor in the calibration.

    Parameters
    ----------
    fitted_slices : list of (T, SSVIParams)
        Ordered by ascending T.  Typically ``SequentialFitResult.fitted_slices``.
    fitted_prev_Ts : list of T | None, optional
        The actual ``prev_T`` used in the calibration for each fitted slice.
        If provided, must have the same length as ``fitted_slices``.
        If ``None``, falls back to using the immediately preceding fitted
        slice (legacy behavior — INCORRECT when there are consecutive
        fallbacks, because ``fit_ssvi_surface_sequential`` does not
        update ``last_valid_prev`` on fallback).
    tol : float
        Numerical tolerance on the inequality checks.

    Returns
    -------
    list of dict, one per fitted slice with a valid ``prev``:
        - "slice_T": float — T of this slice
        - "prev_T": float — T of the actual predecessor
        - "theta_self", "theta_prev": float
        - "theta_ok": bool — True if theta_self >= theta_prev - tol
        - "chi_self", "chi_prev": float — chi = theta * psi
        - "chi_ok": bool — True if chi_self >= chi_prev - tol
        - "rho_chi_self", "rho_chi_prev": float — rho * chi
        - "ratio_value": float or None — |rho_chi_self - rho_chi_prev| /
          (chi_self - chi_prev) when chi genuinely increases
          (chi_delta >= tol); else None (undefined).  A flat or decreasing
          chi makes the ratio denominator <= 0 — the old clamped-to-tol
          denominator manufactured a huge, misleading ratio, so the value
          is now reported as undefined instead.
        - "ratio_ok": bool or None — True if ratio_value <= 1 + tol;
          None when the ratio is undefined (chi not increasing)
        - "failing_conditions": list of str — subset of {"theta", "chi",
          "ratio"}.  "ratio" is listed ONLY when chi increases and the
          slope condition fails — never as a derived consequence of a chi
          dip (that is the "chi" failure).

    Slices with no valid predecessor (``prev_T`` is None or not in the
    fitted-slices dict) are skipped (not included in the output).
    """
    params_by_T: dict[float, SSVIParams] = {T: p for T, p in fitted_slices}
    results: list[dict] = []

    for i, (slice_T, params) in enumerate(fitted_slices):
        # Determine the predecessor T
        if fitted_prev_Ts is not None and i < len(fitted_prev_Ts):
            prev_T = fitted_prev_Ts[i]
        elif i > 0:
            prev_T = fitted_slices[i - 1][0]
        else:
            prev_T = None

        # Skip if no valid predecessor
        if prev_T is None or prev_T not in params_by_T:
            continue

        prev_params = params_by_T[prev_T]

        theta_self = params.theta
        theta_prev = prev_params.theta
        theta_ok = theta_self >= theta_prev - tol

        chi_self = params.theta * params.psi
        chi_prev = prev_params.theta * prev_params.psi
        chi_ok = chi_self >= chi_prev - tol

        rho_chi_self = params.rho * chi_self
        rho_chi_prev = prev_params.rho * chi_prev

        # The ratio condition |(rho*chi)'| / chi' <= 1 is only meaningful
        # when chi genuinely increases.  When chi is flat or decreases the
        # denominator is <= 0; clamping it to `tol` (the old behaviour)
        # manufactured a huge, misleading ratio.  Report the ratio as
        # undefined (None) in that case and never list it as a failing
        # condition — a chi dip is the primary failure, and the ratio is a
        # derived diagnostic, not an independent model violation.
        chi_delta = chi_self - chi_prev
        if chi_delta >= tol:
            ratio_value = abs(rho_chi_self - rho_chi_prev) / chi_delta
            ratio_ok = ratio_value <= 1.0 + tol
        else:
            ratio_value = None
            ratio_ok = None

        failing: list[str] = []
        if not theta_ok:
            failing.append("theta")
        if not chi_ok:
            failing.append("chi")
        if ratio_ok is False:
            failing.append("ratio")

        results.append({
            "slice_T": slice_T,
            "prev_T": prev_T,
            "theta_self": theta_self,
            "theta_prev": theta_prev,
            "theta_ok": theta_ok,
            "chi_self": chi_self,
            "chi_prev": chi_prev,
            "chi_ok": chi_ok,
            "rho_chi_self": rho_chi_self,
            "rho_chi_prev": rho_chi_prev,
            "ratio_value": ratio_value,
            "ratio_ok": ratio_ok,
            "failing_conditions": failing,
        })

    return results

