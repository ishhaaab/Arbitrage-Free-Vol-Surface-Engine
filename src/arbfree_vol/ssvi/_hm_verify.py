"""Post-fit H&M Prop 3.1 / calendar-arbitrage verification helpers."""

import numpy as np
from numpy.typing import NDArray
from arbfree_vol.ssvi.model import SSVIParams, ssvi_w


def verify_hm_condition(
    params_seq: list[SSVIParams],
    *,
    tol: float = 1e-8,
) -> bool:
    """Check the Hendriks-Martini Prop 3.1 no-calendar-spread conditions.

    These parameter conditions are NECESSARY for the absence of
    calendar-spread arbitrage between two eSSVI slices (with
    maturity-dependent rho).  They are treated as sufficient by this
    implementation, but that is NOT established: a documented
    counterexample pair passes all three conditions yet crosses in the
    wings (docs/issues.md, "eSSVI calendar certificate is grid-based").
    The full Hendriks & Martini sufficient statement (Prop 3.5) adds a
    disjunction this code does not enforce.  Until it is implemented,
    ``verify_ssvi_calendar_free`` (the dense-grid check) is the
    load-bearing defense and must accompany any arb-free claim.

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

    Note (a)-(c) are algebraically the single necessary condition
    ``chi_{i+1} >= chi_i * max((1+rho_i)/(1+rho_{i+1}),
    (1-rho_i)/(1-rho_{i+1}))`` split into monotonicity + slope form.

    Reference: Hendriks & Martini (2019), J. Comput. Finance 22(5);
    restated in Corbetta et al. (2019), arXiv:1804.04924, Sec 2.2, and
    Mingone (2022), arXiv:2204.00312, Sec 2.1.
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

    ``verify_hm_condition`` checks the necessary H&M parameter conditions
    (treated by this codebase as sufficient — see its docstring for the
    open-counterexample caveat).  This function is a DISCRETE NUMERIC
    complement / defense-in-depth against optimizer or numerical error:
    it directly evaluates ``w_{i+1}(k) >= w_i(k)`` on a dense
    log-moneyness grid, catching any residual crossing that the
    parameter check's tolerance might leave behind.

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


def _hm_breakdown_entry(
    params: SSVIParams,
    prev_params: SSVIParams,
    slice_T: float,
    prev_T: float,
    tol: float,
) -> dict:
    """Build one H&M Prop 3.1 sub-condition breakdown dict.

    Computes the theta/chi/rho·chi fields and the ratio condition for
    ``params`` relative to ``prev_params``, returning the full dict
    whose keys are pinned by ``verify_hm_condition_breakdown``'s
    docstring.  When ``chi`` does not genuinely increase
    (``chi_delta < tol``) the ratio is reported as undefined (``None``)
    and never listed as a failing condition.
    """
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

    return {
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
    }


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

        results.append(
            _hm_breakdown_entry(params, params_by_T[prev_T], slice_T, prev_T, tol)
        )

    return results
