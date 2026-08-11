"""Mutation-testing regressions.

Targeted tests that kill surviving mutants found by mutmut (run in a
Linux container; mutmut does not run natively on Windows).  These pin
exact formulas / boundary semantics / violation metadata that the
behavioral end-to-end tests only exercise indirectly, so the mutations
otherwise survive.

Each section documents which survivor cluster it kills.
"""

from datetime import date
from math import exp as _exp

from pytest import approx

from arbfree_vol.arbitrage.quote_detect import (
    _check_parity,
    _check_wide_spread,
    _normalize_to_calls,
    _parity_rhs,
)
from arbfree_vol.arbitrage.report import ViolationType
from arbfree_vol.arbitrage.svi_detect import _check_min_variance
from arbfree_vol.models.option import (
    BlackScholesInput,
    OptionContract,
    OptionType,
)
from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface
from arbfree_vol.pricing.black_scholes import price
from arbfree_vol.svi.model import SVIParams

SPOT = 100.0
RISK_FREE = 0.05
DIV_YIELD = 0.0
T = 1.0


def _bs_price(option_type: OptionType, strike: float, sigma: float = 0.2) -> float:
    contract = OptionContract(
        symbol="X", option_type=option_type, strike=strike,
        expiry_date=date(2030, 1, 1),
    )
    model = BlackScholesInput(
        contract=contract, spot=SPOT, expiry_time=T,
        risk_free=RISK_FREE, div_yield=DIV_YIELD, volatility=sigma,
    )
    return price(model)


def _surface(quotes: list[Quote]) -> VolSurface:
    return VolSurface(
        spot=SPOT, risk_free=RISK_FREE, div_yield=DIV_YIELD,
        slices=[ExpirySlice(expiry_time=T, quotes=quotes)],
    )


# ── _parity_rhs: pins S*e^{-qT} - K*e^{-rT} exactly ──────────────────
def test_parity_rhs_exact_value() -> None:
    surface = _surface([
        Quote(strike=100.0, option_type=OptionType.CALL, price=10.0),
    ])
    s = surface.slices[0]
    rhs = _parity_rhs(surface, s, 105.0)
    expected = SPOT * _exp(-DIV_YIELD * T) - 105.0 * _exp(-RISK_FREE * T)
    assert rhs == approx(expected, abs=1e-12)


# ── _normalize_to_calls: pins the parity-implied call arithmetic ──────
def test_normalize_to_calls_exact_values() -> None:
    call = _bs_price(OptionType.CALL, 100.0)
    put = _bs_price(OptionType.PUT, 100.0)
    surface = _surface([
        Quote(strike=100.0, option_type=OptionType.CALL, price=call),
        Quote(strike=100.0, option_type=OptionType.PUT, price=put),
        Quote(strike=110.0, option_type=OptionType.PUT, price=put),
    ])
    s = surface.slices[0]

    # forward_price=None branch (surface-level r/q via _parity_rhs)
    out = _normalize_to_calls(surface, s)
    parity_call_100 = put + SPOT * _exp(-DIV_YIELD * T) - 100.0 * _exp(-RISK_FREE * T)
    expected_110 = put + SPOT * _exp(-DIV_YIELD * T) - 110.0 * _exp(-RISK_FREE * T)
    assert out == approx([
        (100.0, (call + parity_call_100) / 2.0),
        (110.0, expected_110),
    ])

    # forward_price branch: P + e^{-rT}(F - K)
    out_f = _normalize_to_calls(surface, s, forward_price=105.0)
    parity_call_f = put + _exp(-RISK_FREE * T) * (105.0 - 100.0)
    synthetic_110_f = put + _exp(-RISK_FREE * T) * (105.0 - 110.0)
    assert out_f == approx([
        (100.0, (call + parity_call_f) / 2.0),
        (110.0, synthetic_110_f),
    ])


# ── _check_wide_spread: violation metadata + partial bid/ask handling ─
def test_check_wide_spread_violation_metadata() -> None:
    q = Quote(strike=100.0, option_type=OptionType.CALL,
              price=10.0, bid=0.1, ask=0.4)
    violations: list = []
    _check_wide_spread(ExpirySlice(expiry_time=T, quotes=[q]), violations)

    assert len(violations) == 1
    v = violations[0]
    assert v.kind == ViolationType.WIDE_SPREAD
    assert v.magnitude == approx((0.4 - 0.1) / ((0.4 + 0.1) / 2.0))
    assert v.detail is not None
    assert len(v.offending) == 1
    assert v.offending[0].strike == 100.0
    assert v.offending[0].expiry_time == T
    assert v.offending[0].option_type == OptionType.CALL


def test_check_wide_spread_skips_partial_bid_ask() -> None:
    # bid present, ask missing -> skipped, and must not crash.
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=10.0, bid=0.1)
    violations: list = []
    _check_wide_spread(ExpirySlice(expiry_time=T, quotes=[q]), violations)
    assert violations == []


# ── _check_parity: mixed bid/ask must use the fallback threshold ──────
def test_check_parity_partial_bid_ask_uses_fallback() -> None:
    call = _bs_price(OptionType.CALL, 100.0)
    put = _bs_price(OptionType.PUT, 100.0)
    surface = _surface([
        Quote(strike=100.0, option_type=OptionType.CALL,
              price=call, bid=call - 0.5, ask=call + 0.5),
        Quote(strike=100.0, option_type=OptionType.PUT, price=put),
    ])
    violations: list = []
    _check_parity(surface, surface.slices[0], violations)
    # Parity holds with model prices, so no violation — the point is the
    # mixed bid/ask quote must not crash or take the half-spread branch.
    assert violations == []


# ── _check_min_variance: violation metadata for negative w_min ────────
def test_svi_min_variance_violation_fields() -> None:
    params = SVIParams(a=-0.2, b=0.1, rho=0.0, m=0.0, sigma=1.0)
    violations: list = []
    _check_min_variance(params, violations)

    assert len(violations) == 1
    v = violations[0]
    assert v.kind == ViolationType.NEGATIVE_VARIANCE
    assert v.detail is not None
    assert v.offending == ()
    assert v.magnitude > 0


# ── SABR: exact reparametrisation helpers + T != 1 known values ───────
def test_sabr_reparametrisation_roundtrip_exact() -> None:
    """Pins the coefficient<->parameter conversions exactly, so sign /
    floor mutations (+-EPS_FLOOR, clip bounds) cannot survive."""
    from math import atanh, log

    import numpy as np

    from arbfree_vol.sabr.term_structure import (
        EPS_FLOOR,
        _RHO_BOUND,
        _alpha_from_u,
        _nu_from_u,
        _u_from_alpha,
        _u_from_nu,
        _u_from_rho,
    )

    u = np.array([0.0, 1.5, -2.0])
    assert np.allclose(_alpha_from_u(u), np.exp(u) + EPS_FLOOR)
    assert np.allclose(_nu_from_u(u), np.exp(u) + EPS_FLOOR)

    for a in (0.2, 1.0, 5.0):
        assert _u_from_alpha(a) == approx(log(a - EPS_FLOOR))
        assert _u_from_nu(a) == approx(log(a - EPS_FLOOR))

    for rho in (-0.5, 0.0, 0.5):
        expected = atanh(rho / _RHO_BOUND)
        assert _u_from_rho(rho) == approx(expected)

    # extreme rho is clipped by _u_from_rho's safety clip
    assert _u_from_rho(1.0) == approx(atanh(0.99))
    assert _u_from_rho(-1.0) == approx(atanh(-0.99))


def test_sabr_implied_vol_known_values_t_ne_one() -> None:
    """Known-value regression at T != 1: with T == 1 the T-correction
    term ``(1 + corr*T)`` is invariant under the ``*T -> /T`` mutation,
    so T != 1 is required to pin it."""
    from arbfree_vol.sabr.model import sabr_implied_vol

    cases = [
        (2.0, 0.25, 0.065429223887),
        (2.0, -0.25, 0.088904097151),
        (0.5, 0.25, 0.061827977133),
        (0.5, -0.25, 0.084024923727),
    ]
    for T, k, expected in cases:
        iv = sabr_implied_vol(k, 100.0, T, 0.25, 0.5, -0.4, 0.8)
        assert iv == approx(expected, abs=1e-10)


def test_sabr_z_over_x_limit_branch() -> None:
    """Pins the ``|z| < 1e-8`` sub-branch of ``sabr_implied_vol``.

    Mutants ``sabr_implied_vol__mutmut_55`` (``z_over_x = 1.0 -> None``)
    and ``_56`` (``-> 2.0``) survive because every existing test either
    calls with ``|k| < 1e-8`` (top-level ATM guard, returns before the
    branch) or with production-like params where ``(nu/alpha)*F^(1-beta)
    ~ O(1e3)` so ``|z| >= 1e-4`` and the else-branch ``z/x(z)`` runs.

    The extreme ``nu=1e-9`` case drives ``z ~ 1e-15`` while keeping
    ``|k| = 1e-7 >= 1e-8``, so execution reaches the sub-branch: the
    returned value must equal the closed form with ``z_over_x == 1.0``,
    i.e. ``sigma = alpha / FK_pow * (1 + corr * T)``.  With the ``None``
    mutant this call raises TypeError; with the ``2.0`` mutant the value
    is wrong by ~2x, far outside the 1e-12 tolerance.
    """
    from arbfree_vol.sabr.model import sabr_implied_vol

    k, F, T = 1e-7, 100.0, 1.0
    alpha, beta, rho, nu = 1.0, 0.5, -0.4, 1e-9

    # Sanity: |k| must be >= 1e-8, otherwise the top-level ATM guard
    # returns before the |z| sub-branch is ever reached.
    assert abs(k) >= 1e-8

    K = F * _exp(k)
    FK = F * K
    FK_pow = FK ** ((1.0 - beta) / 2.0)
    FK_1mb = FK ** (1.0 - beta)

    # Closed form with z_over_x == 1.0 (Hagan et al. 2002 Eq 2.17a):
    # sigma = alpha / FK_pow * (1 + corr * T), corr the bracketed T term.
    corr = (
        ((1.0 - beta) ** 2 / 24.0) * alpha ** 2 / FK_1mb
        + (rho * beta * alpha * nu) / (4.0 * FK_pow)
        + (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    )
    expected = (alpha / FK_pow) * (1.0 + corr * T)

    got = sabr_implied_vol(k, F, T, alpha, beta, rho, nu)
    assert got == approx(expected, abs=1e-12)


def test_sabr_total_variance_known_value_t_ne_one() -> None:
    """Pins ``w = sigma^2 * T`` (the ``*T -> /T`` mutation is invisible
    at T == 1)."""
    from arbfree_vol.sabr.model import sabr_total_variance

    w = sabr_total_variance(0.25, 100.0, 2.0, 0.25, 0.5, -0.4, 0.8)
    assert w == approx(0.008561966677, abs=1e-10)


def test_calibrate_sabr_default_beta_hint() -> None:
    """The default beta_hint must be 0.5 (a mutation of the default to
    1.5 survives because every existing caller passes beta_hint
    explicitly)."""
    from arbfree_vol.sabr.calibration import calibrate_sabr

    ks = [-0.2, -0.1, 0.0, 0.1, 0.2]
    points = [(k, 0.04 + 0.001 * k) for k in ks]
    fitted = calibrate_sabr(points, forward=100.0, expiry_time=1.0)
    assert fitted.beta == 0.5


# ── Greeks at T != 1, q != 0, PUT: breaks the T/sign equivalences ────
# With T == 1 the mutations ``*sqrt_T -> /sqrt_T`` and ``*T -> /T`` are
# no-ops, and with q == 0 the ``+term3 -> -term3`` theta mutation is a
# no-op.  A PUT also breaks the ``*sign -> /sign`` equivalence.
def test_greeks_known_values_put_t_ne_one() -> None:
    from math import erf, exp, log, sqrt

    from arbfree_vol.models.option import BlackScholesInput, OptionContract
    from arbfree_vol.pricing.greeks import greeks

    S, K, T, r, q, sigma = 100.0, 110.0, 2.0, 0.04, 0.02, 0.3

    def ncdf(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    def npdf(x: float) -> float:
        return exp(-0.5 * x * x) / sqrt(2.0 * 3.141592653589793)

    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    df_q, df_r = exp(-q * T), exp(-r * T)
    n1, n2 = npdf(d1), ncdf(-d1)
    exp_delta = df_q * (ncdf(d1) - 1.0)
    exp_gamma = df_q * n1 / (S * sigma * sqrt(T))
    exp_vega = S * df_q * n1 * sqrt(T)
    exp_theta = -(df_q * S * n1 * sigma) / (2 * sqrt(T)) + r * K * df_r * ncdf(-d2) - q * S * df_q * ncdf(-d1)
    exp_rho = -K * T * df_r * ncdf(-d2)

    contract = OptionContract(
        symbol="X", option_type=OptionType.PUT, strike=K,
        expiry_date=date(2032, 1, 1),
    )
    model = BlackScholesInput(
        contract=contract, spot=S, expiry_time=T,
        risk_free=r, div_yield=q, volatility=sigma,
    )
    got = greeks(model)
    assert got.delta == approx(exp_delta, abs=1e-10)
    assert got.gamma == approx(exp_gamma, abs=1e-10)
    assert got.vega == approx(exp_vega, abs=1e-10)
    assert got.theta == approx(exp_theta, abs=1e-9)
    assert got.rho == approx(exp_rho, abs=1e-10)
