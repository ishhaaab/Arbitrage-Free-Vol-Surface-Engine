"""Gatheral-Jacquier butterfly constraint residuals (GJ 2014, Thm 4.2)."""

import numpy as np
from numpy.typing import NDArray
from arbfree_vol.ssvi.model import _GJ_STRICT_EPS

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
