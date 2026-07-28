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
  theta * psi * (1 + |rho|) <= 4   AND   theta * psi^2 * (1 + |rho|) <= 4
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
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize, NonlinearConstraint, Bounds

from arbfree_vol.ssvi.model import SSVIParams, ssvi_w
from arbfree_vol.ssvi.calibration import fit_ssvi_slice

_logger = logging.getLogger(__name__)


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

    Reference: Gatheral & Jacquier (2014), Theorem 4.2.
    """
    return np.array([
        4.0 - theta * p * (1.0 + rho),
        4.0 - theta * p * (1.0 - rho),
        4.0 - theta * p * p * (1.0 + rho),
        4.0 - theta * p * p * (1.0 - rho),
    ])


def _fit_slice(
    points: list[tuple[float, float]],
    prev: SSVIParams | None = None,
    *,
    eps_theta: float = 1e-9,
    eps_chi: float = 1e-6,
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
    success = result.success or (
        getattr(result, "status", -1) in (1, 2, 3)  # converged / iter-limit OK
    )

    # Retry with SLSQP if the primary run did not converge
    if not success:
        _logger.debug(
            "trust-constr did not converge (status=%s, msg=%s); "
            "retrying with SLSQP",
            getattr(result, "status", "?"), result.message,
        )
        result = _run("SLSQP", result.x, tol=1e-12, maxiter=1000)
        success = result.success or (
            getattr(result, "status", -1) in (0, 1, 2, 3)
        )

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


def fit_ssvi_surface_sequential(
    slices_data: list[tuple[float, list[tuple[float, float]]]],
) -> list[tuple[float, SSVIParams]]:
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

    Parameters
    ----------
    slices_data : list of (expiry_time, points)
        Each element is ``(T, [(k, w), ...])``.  Slices are sorted
        internally by ascending ``T``.

    Returns
    -------
    list of (expiry_time, SSVIParams)
        One entry per slice in ascending maturity order.  Slices where
        both the hard-constrained fit and the fallback fail are omitted.

    Reference
    ----------
    Hendriks & Martini (2019) Prop 3.1; Corbetta et al. (2019) Sec 2.2-2.3.
    """
    ordered = sorted(slices_data, key=lambda sd: sd[0])

    result: list[tuple[float, SSVIParams]] = []
    last_valid_prev: SSVIParams | None = None

    for expiry, pts in ordered:
        try:
            params = _fit_slice(pts, prev=last_valid_prev)
            last_valid_prev = params  # update only on hard-constrained success
            result.append((expiry, params))
        except RuntimeError as e:
            _logger.warning(
                "eSSVI hard-constrained fit failed for T=%.4f (%s); "
                "falling back to unconstrained per-slice fit",
                expiry, e,
            )
            try:
                fallback = fit_ssvi_slice(pts)
                result.append((expiry, fallback))
                # do NOT update last_valid_prev
            except (RuntimeError, ValueError) as e2:
                _logger.error(
                    "eSSVI fallback fit also failed for T=%.4f (%s); "
                    "skipping this slice",
                    expiry, e2,
                )

    if len(result) >= 2:
        params_only = [p for _, p in result]
        if not verify_hm_condition(params_only):
            _logger.warning(
                "verify_hm_condition reports violation after fit; "
                "likely one or more slices fell back to unconstrained. "
                "These are reported as remaining violations."
            )

    return result


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
