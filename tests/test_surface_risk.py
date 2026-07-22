"""Tests for scenario risk and P&L analytics."""

from datetime import date

import pytest
from pytest import approx

from arbfree_vol.models.option import OptionContract, OptionType
from arbfree_vol.surface.interpolate import FittedSurface, iv_at
from arbfree_vol.surface.risk import (
    ScenarioResult,
    parallel_vega_pnl,
    portfolio_pnl,
    spot_bump_analysis,
    vol_bump_analysis,
)
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

_CALL_K100 = OptionContract(
    symbol="X", option_type=OptionType.CALL, strike=100.0,
    expiry_date=_DUMMY_DATE,
)
_PUT_K100 = OptionContract(
    symbol="X", option_type=OptionType.PUT, strike=100.0,
    expiry_date=_DUMMY_DATE,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestPortfolioPnl:
    """Tests for ``portfolio_pnl``."""

    def test_portfolio_pnl_single_call(self) -> None:
        """A single long call gives a positive price."""
        fs = _flat_fs()
        positions = [(_CALL_K100, 0.5, 1.0)]
        pnl = portfolio_pnl(fs, positions)
        # An ATM call should be worth roughly 50% of spot ~ 3-5 at sigma=0.2
        assert pnl > 0.0
        assert pnl < fs.spot

    def test_portfolio_pnl_short_negative(self) -> None:
        """A short position gives negative contribution."""
        fs = _flat_fs()
        positions = [(_CALL_K100, 0.5, -1.0)]
        pnl = portfolio_pnl(fs, positions)
        assert pnl < 0.0

    def test_portfolio_pnl_offsetting(self) -> None:
        """Long call + short call at same strike approx zero P&L."""
        fs = _flat_fs()
        positions = [
            (_CALL_K100, 0.5, 1.0),
            (_CALL_K100, 0.5, -1.0),
        ]
        pnl = portfolio_pnl(fs, positions)
        assert pnl == approx(0.0, abs=1e-12)


class TestSpotBumpAnalysis:
    """Tests for ``spot_bump_analysis``."""

    def test_spot_bump_positive_delta_positive_pnl(self) -> None:
        """Long call: positive spot bump leads to positive P&L."""
        fs = _flat_fs()
        positions = [(_CALL_K100, 0.5, 1.0)]

        results = spot_bump_analysis(fs, positions, bumps=[0.01])
        assert len(results) == 1
        r = results[0]
        assert r.pnl > 0.0
        assert r.spot_bump == 0.01
        assert r.delta_pnl > 0.0
        assert r.portfolio_value_after > r.portfolio_value_before

    def test_spot_bump_long_put_negative_pnl_for_up_move(self) -> None:
        """Long put: positive spot bump leads to negative P&L."""
        fs = _flat_fs()
        positions = [(_PUT_K100, 0.5, 1.0)]

        results = spot_bump_analysis(fs, positions, bumps=[0.01])
        assert len(results) == 1
        r = results[0]
        assert r.pnl < 0.0
        assert r.portfolio_value_after < r.portfolio_value_before

    def test_spot_bump_zero_bump_zero_pnl(self) -> None:
        """A zero spot bump yields zero P&L."""
        fs = _flat_fs()
        positions = [(_CALL_K100, 0.5, 1.0)]

        results = spot_bump_analysis(fs, positions, bumps=[0.0])
        r = results[0]
        assert r.pnl == approx(0.0, abs=1e-12)


class TestVolBumpAnalysis:
    """Tests for ``vol_bump_analysis``."""

    def test_vol_bump_long_call_positive_pnl(self) -> None:
        """Long call is long vega; positive vol shift gives positive P&L."""
        fs = _flat_fs()
        positions = [(_CALL_K100, 0.5, 1.0)]

        results = vol_bump_analysis(fs, positions, vol_shifts=[0.01])
        assert len(results) == 1
        r = results[0]
        assert r.pnl > 0.0

    def test_vol_bump_long_call_negative_shift_negative_pnl(self) -> None:
        """Long call with negative vol shift gives negative P&L."""
        fs = _flat_fs()
        positions = [(_CALL_K100, 0.5, 1.0)]

        results = vol_bump_analysis(fs, positions, vol_shifts=[-0.01])
        assert len(results) == 1
        r = results[0]
        assert r.pnl < 0.0

    def test_vol_bump_zero_shift_zero_pnl(self) -> None:
        """A zero vol shift yields zero P&L."""
        fs = _flat_fs()
        positions = [(_CALL_K100, 0.5, 1.0)]

        results = vol_bump_analysis(fs, positions, vol_shifts=[0.0])
        r = results[0]
        assert r.pnl == approx(0.0, abs=1e-12)


class TestParallelVegaPnl:
    """Tests for ``parallel_vega_pnl``."""

    def test_parallel_vega_pnl_sign(self) -> None:
        """Long call portfolio: positive vega_shift yields positive P&L."""
        fs = _flat_fs()
        positions = [(_CALL_K100, 0.5, 1.0)]

        pnl = parallel_vega_pnl(fs, positions, vega_shift=0.01)
        assert pnl > 0.0

    def test_parallel_vega_pnl_negative_shift(self) -> None:
        """Long call portfolio: negative vega_shift yields negative P&L."""
        fs = _flat_fs()
        positions = [(_CALL_K100, 0.5, 1.0)]

        pnl = parallel_vega_pnl(fs, positions, vega_shift=-0.01)
        assert pnl < 0.0

    def test_parallel_vega_pnl_short_call(self) -> None:
        """Short call (negative vega): positive shift yields negative P&L."""
        fs = _flat_fs()
        positions = [(_CALL_K100, 0.5, -1.0)]

        pnl = parallel_vega_pnl(fs, positions, vega_shift=0.01)
        assert pnl < 0.0

    def test_parallel_vega_pnl_zero_shift(self) -> None:
        """Zero shift yields zero P&L."""
        fs = _flat_fs()
        positions = [(_CALL_K100, 0.5, 1.0)]

        pnl = parallel_vega_pnl(fs, positions, vega_shift=0.0)
        assert pnl == approx(0.0, abs=1e-12)
