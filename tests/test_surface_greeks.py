"""Tests for portfolio-level Greeks computed from a fitted surface."""

from datetime import date

import numpy as np
import pytest
from pytest import approx

from arbfree_vol.models.option import (
    BlackScholesInput,
    OptionContract,
    OptionType,
)
from arbfree_vol.pricing.greeks import greeks as _compute_greeks
from arbfree_vol.surface.greeks import (
    PortfolioGreeks,
    bucketed_greeks,
    portfolio_greeks,
)
from arbfree_vol.surface.interpolate import FittedSurface, iv_at
from arbfree_vol.svi.model import SVIParams
from arbfree_vol.repair.report import FittedSlice

import math


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _forward(T: float, spot: float = 100.0, r: float = 0.05, q: float = 0.0) -> float:
    return spot * math.exp((r - q) * T)


def _flat_fs(
    sigma: float = 0.2,
    spot: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
) -> FittedSurface:
    """One-slice flat fitted surface."""
    T = 0.5
    sl = FittedSlice(
        expiry_time=T,
        params=SVIParams(a=sigma ** 2 * T, b=0.0, rho=0.0, m=0.0, sigma=0.2),
        rmse=0.0,
        forward_price=_forward(T, spot, r, q),
        n_quotes_total=5,
        n_quotes_used=5,
    )
    return FittedSurface(
        spot=spot,
        risk_free=r,
        div_yield=q,
        forward_curve=((T, _forward(T, spot, r, q)),),
        fitted_slices=(sl,),
    )


_DUMMY_DATE = date(2030, 6, 15)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestPortfolioGreeks:
    """Tests for ``portfolio_greeks``."""

    def test_portfolio_greeks_single_call(self) -> None:
        """A portfolio of one call matches the Greeks from a direct
        BlackScholesInput computation."""
        fs = _flat_fs()
        K = fs.spot
        T = 0.5
        contract = OptionContract(
            symbol="X", option_type=OptionType.CALL, strike=K,
            expiry_date=_DUMMY_DATE,
        )
        positions = [(contract, T, 1.0)]

        pg = portfolio_greeks(fs, positions)

        # Expected Greeks from a direct call.
        sigma = iv_at(fs, K, T)
        bsi = BlackScholesInput(
            contract=contract, spot=fs.spot, expiry_time=T,
            risk_free=fs.risk_free, div_yield=fs.div_yield,
            volatility=sigma,
        )
        expected = _compute_greeks(bsi)

        assert pg.total_delta == approx(expected.delta, abs=1e-6)
        assert pg.total_gamma == approx(expected.gamma, abs=1e-6)
        assert pg.total_vega == approx(expected.vega, abs=1e-6)
        assert pg.total_theta == approx(expected.theta, abs=1e-6)
        assert pg.total_rho == approx(expected.rho, abs=1e-6)

    def test_portfolio_greeks_half_position(self) -> None:
        """A half-sized position scales Greeks by 0.5."""
        fs = _flat_fs()
        K = fs.spot
        T = 0.5
        contract = OptionContract(
            symbol="X", option_type=OptionType.CALL, strike=K,
            expiry_date=_DUMMY_DATE,
        )
        positions = [(contract, T, 0.5)]

        pg = portfolio_greeks(fs, positions)

        sigma = iv_at(fs, K, T)
        bsi = BlackScholesInput(
            contract=contract, spot=fs.spot, expiry_time=T,
            risk_free=fs.risk_free, div_yield=fs.div_yield,
            volatility=sigma,
        )
        expected = _compute_greeks(bsi)

        assert pg.total_delta == approx(0.5 * expected.delta, abs=1e-6)
        assert pg.total_gamma == approx(0.5 * expected.gamma, abs=1e-6)

    def test_portfolio_greeks_sign_for_put(self) -> None:
        """A long put has negative delta."""
        fs = _flat_fs()
        contract = OptionContract(
            symbol="X", option_type=OptionType.PUT, strike=fs.spot,
            expiry_date=_DUMMY_DATE,
        )
        positions = [(contract, 0.5, 1.0)]
        pg = portfolio_greeks(fs, positions)
        assert pg.total_delta < 0.0

    def test_portfolio_greeks_short_call_negative_delta(self) -> None:
        """A short call has negative delta exposure."""
        fs = _flat_fs()
        contract = OptionContract(
            symbol="X", option_type=OptionType.CALL, strike=fs.spot,
            expiry_date=_DUMMY_DATE,
        )
        positions = [(contract, 0.5, -1.0)]
        pg = portfolio_greeks(fs, positions)
        assert pg.total_delta < 0.0


class TestPortfolioGreeksEdges:
    """Edge behaviour of ``portfolio_greeks`` (documented in its
    docstring): empty positions, multi-position aggregation, and empty
    surfaces."""

    def test_empty_positions_return_zero_greeks(self) -> None:
        """An empty position list aggregates to zero Greeks."""
        fs = _flat_fs()
        pg = portfolio_greeks(fs, [])
        assert pg.total_delta == 0.0
        assert pg.total_gamma == 0.0
        assert pg.total_vega == 0.0
        assert pg.total_theta == 0.0
        assert pg.total_rho == 0.0

    def test_multi_position_aggregation_matches_analytic_sum(self) -> None:
        """Two positions (long call at K=90, long put at K=110) on the
        flat surface: the aggregate Greeks equal the sum of the
        individual positions' Greeks.  Delta is checked against an
        INDEPENDENT closed-form expectation via math.erf; the remaining
        Greeks against the per-position direct computations."""
        fs = _flat_fs()
        T = 0.5
        r = fs.risk_free
        q = fs.div_yield
        sigma = 0.2
        S = fs.spot

        call = OptionContract(
            symbol="X", option_type=OptionType.CALL, strike=90.0,
            expiry_date=_DUMMY_DATE,
        )
        put = OptionContract(
            symbol="Y", option_type=OptionType.PUT, strike=110.0,
            expiry_date=_DUMMY_DATE,
        )
        positions = [(call, T, 1.0), (put, T, 1.0)]
        pg = portfolio_greeks(fs, positions)

        # Independent closed-form delta:
        #   delta_call = e^{-qT} * N(d1)
        #   delta_put  = -e^{-qT} * N(-d1) = e^{-qT} * (N(d1) - 1)
        def _n_d1(strike: float) -> float:
            d1 = (
                math.log(S / strike) + (r - q + 0.5 * sigma ** 2) * T
            ) / (sigma * math.sqrt(T))
            return 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))

        expected_delta = (
            math.exp(-q * T) * _n_d1(90.0)
            + math.exp(-q * T) * (_n_d1(110.0) - 1.0)
        )
        assert pg.total_delta == approx(expected_delta, abs=1e-9)

        # Per-position sum for the remaining Greeks (aggregation contract).
        expected_gamma = 0.0
        expected_vega = 0.0
        expected_theta = 0.0
        expected_rho = 0.0
        for contract, expiry, qty in positions:
            sigma_pos = iv_at(fs, contract.strike, expiry)
            bsi = BlackScholesInput(
                contract=contract, spot=S, expiry_time=expiry,
                risk_free=r, div_yield=q, volatility=sigma_pos,
            )
            g = _compute_greeks(bsi)
            expected_gamma += qty * g.gamma
            expected_vega += qty * g.vega
            expected_theta += qty * g.theta
            expected_rho += qty * g.rho

        assert pg.total_gamma == approx(expected_gamma, abs=1e-9)
        assert pg.total_vega == approx(expected_vega, abs=1e-9)
        assert pg.total_theta == approx(expected_theta, abs=1e-9)
        assert pg.total_rho == approx(expected_rho, abs=1e-9)

    def test_empty_surface_with_positions_raises(self) -> None:
        """A FittedSurface with no slices cannot supply implied vols:
        iv_at raises ValueError, propagated by portfolio_greeks."""
        fs = FittedSurface(
            spot=100.0,
            risk_free=0.05,
            div_yield=0.0,
            forward_curve=(),
            fitted_slices=(),
        )
        contract = OptionContract(
            symbol="X", option_type=OptionType.CALL, strike=100.0,
            expiry_date=_DUMMY_DATE,
        )
        with pytest.raises(ValueError, match="no slices"):
            portfolio_greeks(fs, [(contract, 0.5, 1.0)])


class TestBucketedGreeks:
    """Tests for ``bucketed_greeks``."""

    def test_bucketed_greeks_shapes(self) -> None:
        """A 3×2 strike/expiry grid returns 5 arrays of shape (3, 2)."""
        fs = _flat_fs()
        strikes = [90.0, 100.0, 110.0]
        expiries = [0.5]

        result = bucketed_greeks(fs, strikes, expiries, OptionType.CALL)

        expected_shape = (3, 1)
        for name in ("delta", "gamma", "vega", "theta", "rho"):
            assert name in result, f"Missing key '{name}'"
            assert result[name].shape == expected_shape, (
                f"Array '{name}' has shape {result[name].shape}, "
                f"expected {expected_shape}"
            )

    def test_bucketed_greeks_two_expiries(self) -> None:
        """A 2×2 grid returns arrays of shape (2, 2)."""
        # Build a two-slice surface so both 0.5 and 1.0 are in range.
        sigma = 0.2
        spot = 100.0
        r = 0.05
        q = 0.0

        def _fwd(T: float) -> float:
            return spot * math.exp((r - q) * T)

        sl1 = FittedSlice(
            expiry_time=0.5,
            params=SVIParams(a=sigma ** 2 * 0.5, b=0.0, rho=0.0, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=_fwd(0.5),
            n_quotes_total=5,
            n_quotes_used=5,
        )
        sl2 = FittedSlice(
            expiry_time=1.0,
            params=SVIParams(a=sigma ** 2 * 1.0, b=0.0, rho=0.0, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=_fwd(1.0),
            n_quotes_total=5,
            n_quotes_used=5,
        )
        fs = FittedSurface(
            spot=spot,
            risk_free=r,
            div_yield=q,
            forward_curve=((0.5, _fwd(0.5)), (1.0, _fwd(1.0))),
            fitted_slices=(sl1, sl2),
        )

        strikes = [95.0, 105.0]
        expiries = [0.5, 1.0]

        result = bucketed_greeks(fs, strikes, expiries, OptionType.PUT)

        for name in ("delta", "gamma", "vega", "theta", "rho"):
            assert result[name].shape == (2, 2)

    def test_bucketed_greeks_call_vs_put_gamma_vega_match(self) -> None:
        """Gamma and vega should be identical for call and put at same
        (K, T)."""
        fs = _flat_fs()
        strikes = [100.0]
        expiries = [0.5]

        call_result = bucketed_greeks(fs, strikes, expiries, OptionType.CALL)
        put_result = bucketed_greeks(fs, strikes, expiries, OptionType.PUT)

        assert call_result["gamma"][0, 0] == approx(
            put_result["gamma"][0, 0], abs=1e-10
        )
        assert call_result["vega"][0, 0] == approx(
            put_result["vega"][0, 0], abs=1e-10
        )
