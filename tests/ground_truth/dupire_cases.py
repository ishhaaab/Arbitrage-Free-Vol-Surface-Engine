"""Dupire local-volatility ground-truth cases.

Three closed-form surfaces:

1. CONSTANT-VOL BS: w(k, T) = sigma^2 * T  ->  local vol = sigma everywhere.
2. LINEAR-IN-K:      w(k, T) = sigma^2 * T * (1 + beta * k) with an
   analytically-derived local vol.
3. NON-FLAT SVI:     the two-slice surface ``tests/test_local_vol.py`` pins
   (a = a_ref*T, b = b_ref*T, rho/m/sigma fixed, r = 0.05, q = 0), whose
   local vol is evaluated by an independent closed form
   (``closed_form_nonflat_svi_sigma_loc``) using the surface's analytic SVI
   derivatives and the exact fixed-k ∂w/∂T — the executable replacement for
   the previously opaque literals in the repo's regression test.

The linear-in-k closed form is implemented LITERALLY below, independent of
the repo's ``local_vol.py``.  Derivation (Gatheral 2004, Eq 1.10 — the
formula the repo's ``local_vol.py`` implements; see its docstring):

    sigma_loc^2(K, T) = dw/dT / [ 1 - (k/w) dw/dk
                                  + (1/4)(-1/4 - 1/w + k^2/w^2) (dw/dk)^2
                                  + (1/2) d2w/dk2 ]

with w = w(k, T), k = log(K/F(T)).  For w = sigma^2 T (1 + beta k):

    dw/dT  = sigma^2 (1 + beta k)
    dw/dk  = sigma^2 T beta
    d2w/dk2 = 0

so

    sigma_loc^2 = sigma^2 (1 + beta k)
                  / [ 1 - (k/w) sigma^2 T beta
                      + (1/4)(-1/4 - 1/w + k^2/w^2) (sigma^2 T beta)^2 ]

.. note::

   The task brief quoted the simplified denominator term
   ``(1/4)(-1/4 + 1/w) (dw/dk)^2`` (no ``k^2/w^2``, flipped ``1/w`` sign).
   That is NOT the repo's formula and does not match the independent
   price-space Dupire finite-difference ground truth (verified numerically:
   the brief's form deviates by ~5e-4 in sigma for beta=0.3, while the
   Gatheral form agrees with price-space FD to ~1e-4).  The closed form
   implemented here is the CORRECT Gatheral form, matching both the book
   (Chapter 7, recall of Eq 1.10) and price-space Dupire FD.

Construction of the repo-consumable ``FittedSurface`` for the linear case
-----------------------------------------------------------------------
A raw-SVI slice cannot be exactly linear in k on both sides of its kink
(below ``m`` the slope changes sign).  We place the kink far below the
evaluation grid (``m = -1.0``, grid k in [-0.5, 0.5]) with a tiny smoothing
``sigma_param = 1e-3``, so on the grid the SVI smile equals the linear form
up to ``b * sigma_param^2 / (2 |k-m|) ~ 1e-8`` in total variance — two
orders of magnitude below the finite-difference tolerance of the test.
Slices scale ``a, b`` linearly with ``T`` (``a = sigma^2 T (1 + beta m)``,
``b = sigma^2 T beta``), so w(k, T) is exactly linear in T for fixed k and
``dw/dT = sigma^2 (1 + beta k)`` holds at every interior maturity.

The constant-vol surface follows the repo's own ``test_local_vol.py``
convention (``a = sigma^2 * T``, ``b = 0``, ``rho = m = 0``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.surface.interpolate import FittedSurface, _forward_at
from arbfree_vol.svi.model import SVIParams

# Surface parameters (documented, not magic numbers).
SIGMA_CONST: float = 0.25            # constant-vol case sigma
SIGMA_LIN: float = 0.25              # linear-in-k case sigma0
BETA_LIN: float = 0.3                # linear-in-k case beta
M_LIN: float = -1.0                  # SVI kink placement (below the grid)
S_LIN: float = 1e-3                  # SVI smoothing (branch linearity ~1e-8)
RHO_LIN: float = 0.0                 # SVI rho for the linear branch
SPOT: float = 100.0                  # r = q = 0 -> forward = spot constant
RISK_FREE: float = 0.0
DIV_YIELD: float = 0.0
MATURITIES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)

# FD tolerance documentation
FD_REL_TOL_CONST: float = 1e-9
"""Relative tolerance for the constant-vol case.  The flat smile has
``dw/dk = d2w/dk2 = 0`` and w linear in T, so every finite difference is
exact and the measured deviation is ~1e-14; 1e-9 leaves room for platform
float noise while still being machine-tight."""

FD_REL_TOL_LINEAR: float = 2e-3
"""Relative tolerance for the linear-in-k case.  Finite differences are
first/second-order accurate in the k-step (the repo's FD steps are
``dK = max(1e-3, K*1e-3)``); the independent price-space FD reference
agrees with the closed form to ~1.5e-4, so 2e-3 is a documented FD
tolerance several times the observed FD error.  (The repo previously
FAILED this tolerance by 4e-3..8e-3 because of the ``_d2w_dk2``
first-derivative bias; after the non-uniform stencil + fixed-k ``_dw_dT``
fix the measured deviation is ~3.4e-7 — see test_dupire_ground_truth.py.)"""

# Non-flat SVI surface parameters (the surface the repo's
# ``test_local_vol.py::TestDupireNonFlatSmileExact`` pins).  ``a`` and ``b``
# scale linearly with T while rho/m/sigma_param stay fixed, so the smile
# steepens as T grows; with r != q the per-slice forwards differ and the
# interpolated surface is NOT exactly w = T*f(k).
NONFLAT_SPOT: float = 100.0
NONFLAT_RISK_FREE: float = 0.05
NONFLAT_DIV_YIELD: float = 0.0
NONFLAT_T_LOW: float = 0.5
NONFLAT_T_HIGH: float = 2.0
NONFLAT_A_REF: float = 0.04
NONFLAT_B_REF: float = 0.4
NONFLAT_RHO_REF: float = -0.4
NONFLAT_M_REF: float = 0.05
NONFLAT_SIGMA_REF: float = 0.15

# The six (K, T) interior points the repo regression test asserts on —
# interior in both dimensions (away from the slice expiries and the FD
# stencil edges), per the repo's own guidance for clean Dupire values.
NONFLAT_REFERENCE_POINTS: tuple[tuple[float, float], ...] = (
    (90.0, 1.0),
    (90.0, 1.5),
    (100.0, 1.0),
    (100.0, 1.5),
    (110.0, 1.0),
    (110.0, 1.5),
)

FD_REL_TOL_NONFLAT: float = 1e-4
"""Relative tolerance for the non-flat SVI surface.  The repo's
total-variance-space FD stencil (dK = max(1e-3, K*1e-3), dT = 1e-3) is
first/second-order accurate in the k-step; the measured deviation of the
repo's ``dupire`` vs the independent closed form is ~2-3.5e-6 relative, so
1e-4 is a documented FD tolerance ~30x the observed error.  (The OLD
symmetric ``_d2w_dk2`` stencil failed this surface by ~12% relative at
K=90/T=1.0 — the fixed non-uniform stencil removed that bias.)"""


def _forward(T: float) -> float:
    return SPOT * math.exp((RISK_FREE - DIV_YIELD) * T)


def _linear_slice_params(T: float) -> tuple[float, float]:
    """(a, b) of the SVI slice that realises w = sigma^2 T (1 + beta k)."""
    a = SIGMA_LIN ** 2 * T * (1.0 + BETA_LIN * M_LIN)
    b = SIGMA_LIN ** 2 * T * BETA_LIN
    return a, b


def build_constant_vol_surface() -> FittedSurface:
    """Two-plus-slice flat surface: w(k, T) = SIGMA_CONST^2 * T."""
    slices = tuple(
        FittedSlice(
            expiry_time=T,
            params=SVIParams(a=SIGMA_CONST ** 2 * T, b=0.0, rho=0.0,
                             m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=_forward(T),
            n_quotes_total=5,
            n_quotes_used=5,
        )
        for T in MATURITIES
    )
    return FittedSurface(
        spot=SPOT,
        risk_free=RISK_FREE,
        div_yield=DIV_YIELD,
        forward_curve=tuple((T, _forward(T)) for T in MATURITIES),
        fitted_slices=slices,
    )


def build_linear_in_k_surface() -> FittedSurface:
    """Surface whose total variance is (on the grid) sigma^2 T (1 + beta k).

    The SVI branch construction is documented in the module docstring; the
    deviation from exact linearity is ~1e-8 in total variance.
    """
    slices = tuple(
        FittedSlice(
            expiry_time=T,
            params=SVIParams(a=_linear_slice_params(T)[0],
                             b=_linear_slice_params(T)[1],
                             rho=RHO_LIN, m=M_LIN, sigma=S_LIN),
            rmse=0.0,
            forward_price=_forward(T),
            n_quotes_total=5,
            n_quotes_used=5,
        )
        for T in MATURITIES
    )
    return FittedSurface(
        spot=SPOT,
        risk_free=RISK_FREE,
        div_yield=DIV_YIELD,
        forward_curve=tuple((T, _forward(T)) for T in MATURITIES),
        fitted_slices=slices,
    )


def build_nonflat_svi_surface() -> FittedSurface:
    """Two-slice NON-FLAT SVI surface (the repo's test_local_vol.py pin).

    ``a`` and ``b`` scale linearly with T (``a = a_ref*T``, ``b = b_ref*T``)
    while ``rho, m, sigma_param`` stay fixed, so the smile steepens as T
    grows.  With ``r = 0.05 != q = 0`` the per-slice forwards differ, so
    the interpolated surface is NOT exactly ``w = T*f(k)``: the repo's
    ``total_variance_at`` evaluates each slice at ITS OWN forward's
    log-moneyness and linearly interpolates the total variances in T.
    This is the surface the closed-form evaluator below (and the repo's
    regression test) operate on.
    """
    slices = tuple(
        FittedSlice(
            expiry_time=T,
            params=SVIParams(a=NONFLAT_A_REF * T, b=NONFLAT_B_REF * T,
                             rho=NONFLAT_RHO_REF, m=NONFLAT_M_REF,
                             sigma=NONFLAT_SIGMA_REF),
            rmse=0.0,
            forward_price=NONFLAT_SPOT * math.exp(
                (NONFLAT_RISK_FREE - NONFLAT_DIV_YIELD) * T
            ),
            n_quotes_total=5,
            n_quotes_used=5,
        )
        for T in (NONFLAT_T_LOW, NONFLAT_T_HIGH)
    )
    return FittedSurface(
        spot=NONFLAT_SPOT,
        risk_free=NONFLAT_RISK_FREE,
        div_yield=NONFLAT_DIV_YIELD,
        forward_curve=tuple(
            (T, NONFLAT_SPOT * math.exp((NONFLAT_RISK_FREE - NONFLAT_DIV_YIELD) * T))
            for T in (NONFLAT_T_LOW, NONFLAT_T_HIGH)
        ),
        fitted_slices=slices,
    )


# ---------------------------------------------------------------------------
# Independent literal closed forms (anti-circularity)
# ---------------------------------------------------------------------------

def closed_form_linear_sigma_loc(k: float, T: float) -> float:
    """Analytic local vol for w = sigma^2 T (1 + beta k).

    Gatheral (2004) Eq 1.10 with dw/dT = sigma^2(1+beta k),
    dw/dk = sigma^2 T beta, d2w/dk2 = 0.  Returns the positive square root.
    """
    w = SIGMA_LIN ** 2 * T * (1.0 + BETA_LIN * k)
    dwdT = SIGMA_LIN ** 2 * (1.0 + BETA_LIN * k)
    dwdk = SIGMA_LIN ** 2 * T * BETA_LIN
    denominator = (
        1.0
        - (k / w) * dwdk
        + 0.25 * (-0.25 - 1.0 / w + k * k / (w * w)) * (dwdk * dwdk)
    )
    return math.sqrt(dwdT / denominator)


def _svi_literal_w(k: float, a: float, b: float, rho: float, m: float,
                   sigma: float) -> tuple[float, float, float]:
    """Literal raw-SVI total variance and its first/second k-derivatives.

    ``w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))`` with
    ``w'(k) = b*(rho + (k-m)/sqrt((k-m)^2 + sigma^2))`` and
    ``w''(k) = b*sigma^2 / ((k-m)^2 + sigma^2)^(3/2)``.  These are the
    repo's ``svi_core`` formulas implemented LITERALLY here so the
    closed-form evaluator never calls a repo helper.
    """
    u = k - m
    r = math.sqrt(u * u + sigma * sigma)
    w = a + b * (rho * u + r)
    w1 = b * (rho + u / r)
    w2 = b * sigma * sigma / (r ** 3)
    return w, w1, w2


def closed_form_nonflat_svi_sigma_loc(fs: FittedSurface, K: float, T: float) -> float:
    """Independent closed-form local vol on a piecewise-linear SVI surface.

    Gatheral (2004) Eq 1.10 evaluated EXACTLY on the repo's interpolated
    surface semantics (``total_variance_at``): each slice is evaluated at
    ITS OWN forward's log-moneyness (``k_i = ln(K/F_i)`` with the per-slice
    ``F_i``) and the total variances are linearly interpolated in T between
    the bracketing slices.  The k-derivatives are the analytic raw-SVI
    derivatives of each slice (``_svi_literal_w``) weighted by the
    interpolation fraction; the T-derivative is the exact fixed-k ∂w/∂T
    (the repo's ``_dw_dT`` convention, but analytic): along
    ``k = ln(K/F(T))`` the strike re-strikes with the forward (``K ∝ F(T)``),
    giving ``∂w/∂T = (w_high - w_low)/(T_high - T_low) + mu * ∂w/∂k`` with
    ``mu = F'(T)/F(T)`` — the repo interpolates the forward LINEARLY in T,
    so ``mu`` is not ``r-q`` in general (the FIX-GT corrected convention).

    Returns the positive square root (``nan`` if the denominator or the
    variance is non-positive).  Targets INTERIOR maturities (strictly
    between slice expiries), matching the repo's advice to keep the FD
    stencil away from the kinks at slice boundaries.
    """
    slices = fs.fitted_slices
    n = len(slices)
    if n < 2:
        raise ValueError("closed_form_nonflat_svi_sigma_loc needs >= 2 slices")
    i_low, i_high = 0, n - 1
    for i in range(n - 1):
        if slices[i].expiry_time <= T <= slices[i + 1].expiry_time:
            i_low, i_high = i, i + 1
            break
    sl_low, sl_high = slices[i_low], slices[i_high]
    T_low = sl_low.expiry_time
    T_high = sl_high.expiry_time
    dT = T_high - T_low
    theta = (T - T_low) / dT

    F_T = _forward_at(fs, T)
    F_low = _forward_at(fs, T_low)
    F_high = _forward_at(fs, T_high)
    k = math.log(K / F_T)
    k_low = math.log(K / F_low)
    k_high = math.log(K / F_high)

    p_low, p_high = sl_low.params, sl_high.params
    wA, wA1, wA2 = _svi_literal_w(k_low, p_low.a, p_low.b, p_low.rho,
                                  p_low.m, p_low.sigma)
    wB, wB1, wB2 = _svi_literal_w(k_high, p_high.a, p_high.b, p_high.rho,
                                  p_high.m, p_high.sigma)

    w = (1.0 - theta) * wA + theta * wB
    dwdk = (1.0 - theta) * wA1 + theta * wB1
    d2wdk2 = (1.0 - theta) * wA2 + theta * wB2
    mu = (F_high - F_low) / (dT * F_T)  # F'(T)/F(T), F linear in T
    dwdT = (wB - wA) / dT + mu * dwdk

    denominator = (
        1.0
        - (k / w) * dwdk
        + 0.25 * (-0.25 - 1.0 / w + k * k / (w * w)) * (dwdk * dwdk)
        + 0.5 * d2wdk2
    )
    if denominator <= 0.0:
        return float("nan")
    sigma_loc_sq = dwdT / denominator
    if sigma_loc_sq <= 0.0:
        return float("nan")
    return math.sqrt(sigma_loc_sq)


def price_space_dupire_sigma_loc(
    fs: FittedSurface, K: float, T: float,
    dT: float = 1e-4, dK_rel: float = 1e-2,
) -> float:
    """Independent price-space Dupire local vol via finite differences.

    Computes sigma_loc^2 = [dC/dT + mu*K dC/dK + (r - mu) C] / ((1/2) K^2 C_KK)
    directly from Black-Scholes call prices built on ``total_variance_at``
    — the repo's ACTUAL surface — using the repo's OWN forward curve
    (``_forward_at``) and the CORRECTED forward drift ``mu = F'(T)/F(T)``
    (the repo interpolates the forward LINEARLY in T, so ``mu`` is not
    ``r-q`` in general — the FIX-GT corrected convention; when the forward
    IS ``F = S*exp((r-q)T)`` the formula reduces to the standard
    ``[dC/dT + (r-q)K dC/dK + qC]`` form).  This is the model-independent
    definition of local volatility and does not use any total-variance-space
    formula, so it is a fully independent ground truth for the repo's
    surface.
    """
    from scipy.stats import norm

    from arbfree_vol.surface.interpolate import _forward_at, total_variance_at

    r = fs.risk_free
    _MU_H: float = 1e-4  # central-difference step for mu = F'/F

    def forward(T0: float) -> float:
        return _forward_at(fs, T0)

    def mu(T0: float) -> float:
        """F'(T0)/F(T0) via a central difference of ln F on the actual curve."""
        return (math.log(forward(T0 + _MU_H)) - math.log(forward(T0 - _MU_H))) / (2.0 * _MU_H)

    def bs_price(K0: float, T0: float) -> float:
        F0 = forward(T0)
        w = total_variance_at(fs, K0, T0)
        sigma = math.sqrt(w / T0)
        s = sigma * math.sqrt(T0)
        d1 = (math.log(F0 / K0) + 0.5 * s * s) / s
        d2 = d1 - s
        return math.exp(-r * T0) * (F0 * norm.cdf(d1) - K0 * norm.cdf(d2))

    dK = dK_rel * K
    C = bs_price(K, T)
    dC_dT = (bs_price(K, T + dT) - bs_price(K, T - dT)) / (2.0 * dT)
    dC_dK = (bs_price(K + dK, T) - bs_price(K - dK, T)) / (2.0 * dK)
    CKK = (bs_price(K + dK, T) - 2.0 * C + bs_price(K - dK, T)) / (dK * dK)
    numerator = dC_dT + mu(T) * K * dC_dK + (r - mu(T)) * C
    sigma_loc_sq = numerator / (0.5 * K * K * CKK)
    if sigma_loc_sq <= 0.0:
        return float("nan")
    return math.sqrt(sigma_loc_sq)


# ---------------------------------------------------------------------------
# Grid definitions
# ---------------------------------------------------------------------------

CONSTANT_VOL_STRIKES: tuple[float, ...] = tuple(
    SPOT * math.exp(k) for k in (-0.2, -0.1, 0.0, 0.1, 0.2)
)
CONSTANT_VOL_MATURITIES: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)
# Interior maturity rows (strictly inside the surface range, away from the
# FD stencil edges) for the constant-vol assertion.
CONSTANT_VOL_INTERIOR_ROWS: tuple[int, ...] = (1, 2, 3, 4, 5)

LINEAR_IN_K_STRIKES: tuple[float, ...] = tuple(
    SPOT * math.exp(k) for k in (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0,
                                 0.1, 0.2, 0.3, 0.4, 0.5)
)
LINEAR_IN_K_MATURITIES: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5)
LINEAR_IN_K_INTERIOR_ROWS: tuple[int, ...] = (1, 2, 3, 4)
