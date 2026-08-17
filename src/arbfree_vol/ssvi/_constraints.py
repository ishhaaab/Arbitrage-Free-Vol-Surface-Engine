"""Hard-constraint builders and the constrained optimizer for eSSVI fits.

Extracted from ``term_structure._fit_slice`` so the H&M Prop 3.1
constraint math and the trust-constr → SLSQP retry are unit-testable in
isolation and the sequential-fit module stays a thinner orchestrator.

``minimize_fn`` is a parameter of ``_constrained_minimize`` (rather than
an import inside this module) so callers pass their own ``minimize``
binding — the eSSVI tests patch ``term_structure.minimize`` to script
optimizer statuses, and that patch must keep working.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, NonlinearConstraint, minimize, OptimizeResult

from arbfree_vol.ssvi.model import SSVIParams
from arbfree_vol.ssvi._butterfly import _butterfly_constraints

_logger = logging.getLogger(__name__)


def _hard_constraints(
    prev: SSVIParams | None,
    eps_theta: float,
    eps_chi: float,
) -> list[NonlinearConstraint]:
    """All hard no-arbitrage constraints for one slice fit.

    Always adds the per-slice butterfly constraints (four ``>= 0``
    residuals via ``_butterfly_constraints``).  When ``prev`` is given,
    adds the Hendriks & Martini (2019) Prop 3.1 calendar constraints:

    (a) theta non-decreasing:      theta - theta_prev >= eps_theta
    (b) chi non-decreasing:        theta*psi - chi_prev >= eps_chi
    (c) ratio bound, as two linear-fractional inequalities:
        |rho*chi - rho_prev*chi_prev| / (chi - chi_prev) <= 1,
        written as ``_ratio_upper``/``_ratio_lower`` in ``[-1, 1]``.

    The constraints are expressed in the optimizer's unconstrained
    parameterization ``(theta, u = arctanh(rho), v = log(psi))``.
    """
    constraints: list[NonlinearConstraint] = []

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

    return constraints


def _constrained_minimize(
    objective,
    x0: NDArray[np.float64],
    bounds: Bounds,
    constraints: list[NonlinearConstraint],
    minimize_fn=minimize,
) -> OptimizeResult:
    """Minimize with hard constraints, retrying trust-constr with SLSQP.

    Primary attempt is ``trust-constr``; on failure the result is retried
    with ``SLSQP``.  Only ``result.success`` is trusted for convergence:

    - ``trust-constr``: success is True exactly for statuses 1/2 (gtol /
      xtol satisfied).  Status 0 (max f-evals), 3 (callback termination —
      no callback is ever passed here) and 4 ("minimize successful but
      constraints not satisfied") are failures.
    - ``SLSQP``: success is True ONLY for exit mode 0 ("Optimization
      terminated successfully").  Modes 1 (stalled line search), 2
      (degenerate problem) and 3 (LSQ-subproblem iteration cap) are NOT
      convergence — accepting them would certify a non-converged fit as
      hard-constrained arb-free and skip the fallback bookkeeping.

    ``minimize_fn`` defaults to scipy's ``minimize`` and is injectable so
    the eSSVI tests can script optimizer statuses by patching the
    caller's ``minimize`` binding.

    Raises ``RuntimeError`` if both attempts fail to converge.
    """
    def _run(method: str, x_init, tol: float, maxiter: int):
        opts: dict = {"maxiter": maxiter}
        if method == "trust-constr":
            opts["gtol"] = tol
        else:  # SLSQP
            opts["ftol"] = tol
        return minimize_fn(
            objective,
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

    return result
