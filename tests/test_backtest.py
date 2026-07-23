"""Tests for the mispricing backtest engine (Wave 1)."""

from __future__ import annotations

import math
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from pytest import approx

from arbfree_vol.models.option import OptionContract, OptionType
from arbfree_vol.models.surface import (
    VolSurface,
    ExpirySlice,
    Quote,
    get_r,
    get_q,
)
from arbfree_vol.pricing.black_scholes import price_floats
from arbfree_vol.pricing.implied_vol import implied_vol
from arbfree_vol.svi.model import SVIParams
from arbfree_vol.repair.report import FittedSlice
from arbfree_vol.surface.interpolate import FittedSurface, iv_at

from arbfree_vol.backtest.types import (
    MispricingSignal,
    Trade,
    TradePnL,
    BacktestResult,
)
from arbfree_vol.backtest.signal import detect_mispricing
from arbfree_vol.backtest.pnl import realize_trade_pnl
from arbfree_vol.backtest.engine import run_backtest
from arbfree_vol.backtest.prices import fetch_underlying_path

# ---------------------------------------------------------------------------
# Helpers (matching test_surface_risk.py style)
# ---------------------------------------------------------------------------

_DUMMY_DATE = date(2030, 6, 15)
"""Fixed snapshot date for all tests (no date-dependence)."""


def _forward(T: float, spot: float = 100.0, r: float = 0.05, q: float = 0.0) -> float:
    return spot * math.exp((r - q) * T)


def _flat_fs(
    sigma: float = 0.2,
    spot: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
    T: float = 0.5,
) -> FittedSurface:
    """Single-slice flat fitted surface at constant implied vol."""
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


def _flat_fs_multi(
    expiries: list[float],
    sigma: float = 0.2,
    spot: float = 100.0,
    r: float = 0.05,
    q: float = 0.0,
) -> FittedSurface:
    """Multi-slice flat fitted surface."""
    slices: list[FittedSlice] = []
    fwd_curve: list[tuple[float, float]] = []
    for T in expiries:
        sl = FittedSlice(
            expiry_time=T,
            params=SVIParams(a=sigma ** 2 * T, b=0.0, rho=0.0, m=0.0, sigma=0.2),
            rmse=0.0,
            forward_price=_forward(T, spot, r, q),
            n_quotes_total=5,
            n_quotes_used=5,
        )
        slices.append(sl)
        fwd_curve.append((T, _forward(T, spot, r, q)))
    return FittedSurface(
        spot=spot,
        risk_free=r,
        div_yield=q,
        forward_curve=tuple(sorted(fwd_curve, key=lambda x: x[0])),
        fitted_slices=tuple(sorted(slices, key=lambda s: s.expiry_time)),
    )


def _flat_path(
    start: date,
    end: date,
    price: float = 100.0,
) -> dict[date, float]:
    """Deterministic flat price path."""
    path: dict[date, float] = {}
    d = start
    while d <= end:
        path[d] = price
        d += timedelta(days=1)
    return path


def _linear_path(
    start: date,
    end: date,
    start_price: float = 100.0,
    end_price: float = 110.0,
) -> dict[date, float]:
    """Monotonic linear ramp from start_price to end_price."""
    n_days = (end - start).days
    path: dict[date, float] = {}
    for i in range(n_days + 1):
        d = start + timedelta(days=i)
        frac = i / n_days if n_days > 0 else 0.0
        path[d] = start_price + frac * (end_price - start_price)
    return path


# ===================================================================
#  Signal detection tests
# ===================================================================

class TestDetectMispricing:
    """Tests for ``detect_mispricing``."""

    def _make_quote_surface(
        self,
        T: float,
        quotes: list[Quote],
        spot: float = 100.0,
        r: float = 0.05,
        q: float = 0.0,
    ) -> VolSurface:
        return VolSurface(
            spot=spot,
            risk_free=r,
            div_yield=q,
            slices=[ExpirySlice(expiry_time=T, quotes=quotes)],
        )

    # ------------------------------------------------------------------
    def test_finds_overpriced(self) -> None:
        """Quote with IV > model IV + threshold → side = -1, mispricing > 0."""
        fs = _flat_fs(sigma=0.2, T=0.5)
        price_over = price_floats(100.0, 100.0, 0.5, 0.05, 0.0, 0.25, True)
        surface = self._make_quote_surface(
            0.5, [Quote(strike=100.0, option_type=OptionType.CALL, price=price_over)]
        )
        signals = detect_mispricing(surface, fs, threshold=0.01, snapshot_date=_DUMMY_DATE)
        assert len(signals) == 1
        sg = signals[0]
        assert sg.side == -1
        assert sg.mispricing > 0.0
        assert sg.market_iv == approx(0.25, abs=1e-4)

    # ------------------------------------------------------------------
    def test_finds_underpriced(self) -> None:
        """Quote with IV < model IV - threshold → side = +1, mispricing < 0."""
        fs = _flat_fs(sigma=0.2, T=0.5)
        price_under = price_floats(100.0, 100.0, 0.5, 0.05, 0.0, 0.15, True)
        surface = self._make_quote_surface(
            0.5, [Quote(strike=100.0, option_type=OptionType.CALL, price=price_under)]
        )
        signals = detect_mispricing(surface, fs, threshold=0.01, snapshot_date=_DUMMY_DATE)
        assert len(signals) == 1
        sg = signals[0]
        assert sg.side == +1
        assert sg.mispricing < 0.0
        assert sg.market_iv == approx(0.15, abs=1e-4)

    # ------------------------------------------------------------------
    def test_skips_below_threshold(self) -> None:
        """Quote with |mispricing| <= threshold → no signal."""
        fs = _flat_fs(sigma=0.2, T=0.5)
        # Price at σ = 0.201 → mispricing ≈ 0.001 < 0.01 threshold
        price_near = price_floats(100.0, 100.0, 0.5, 0.05, 0.0, 0.201, True)
        surface = self._make_quote_surface(
            0.5, [Quote(strike=100.0, option_type=OptionType.CALL, price=price_near)]
        )
        signals = detect_mispricing(surface, fs, threshold=0.01, snapshot_date=_DUMMY_DATE)
        assert len(signals) == 0

    # ------------------------------------------------------------------
    def test_skips_near_expiry(self) -> None:
        """Slice with T < min_signal_T_years → skipped even if mispriced."""
        far_T = 0.5
        near_T = 0.02  # ~7 days, below 14/365 ≈ 0.038
        fs = _flat_fs_multi([far_T, near_T], sigma=0.2)

        price_far_over = price_floats(100.0, 100.0, far_T, 0.05, 0.0, 0.25, True)
        price_near_over = price_floats(100.0, 100.0, near_T, 0.05, 0.0, 0.25, True)

        surface = VolSurface(
            spot=100.0, risk_free=0.05, div_yield=0.0,
            slices=[
                ExpirySlice(expiry_time=far_T, quotes=[
                    Quote(strike=100.0, option_type=OptionType.CALL, price=price_far_over),
                ]),
                ExpirySlice(expiry_time=near_T, quotes=[
                    Quote(strike=100.0, option_type=OptionType.CALL, price=price_near_over),
                ]),
            ],
        )
        # min_signal_T_years > near_T but < far_T
        signals = detect_mispricing(
            surface, fs, threshold=0.01,
            min_signal_T_years=14.0 / 365.0,
            snapshot_date=_DUMMY_DATE,
        )
        # Only the far slice should produce a signal
        assert len(signals) == 1
        assert signals[0].expiry_time == approx(far_T)

    # ------------------------------------------------------------------
    def test_skips_unfitted_slice(self) -> None:
        """Slice expiry not in fs.fitted_slices → skipped (no iv_at crash)."""
        fs = _flat_fs(T=0.5)  # only T=0.5
        price = price_floats(100.0, 100.0, 0.7, 0.05, 0.0, 0.25, True)
        surface = VolSurface(
            spot=100.0, risk_free=0.05, div_yield=0.0,
            slices=[
                ExpirySlice(expiry_time=0.5, quotes=[
                    Quote(strike=100.0, option_type=OptionType.CALL, price=price),
                ]),
                ExpirySlice(expiry_time=0.7, quotes=[
                    Quote(strike=100.0, option_type=OptionType.CALL, price=price),
                ]),
            ],
        )
        # Should not raise ValueError from iv_at for the unfitted 0.7 slice
        signals = detect_mispricing(surface, fs, threshold=0.01, snapshot_date=_DUMMY_DATE)
        # Only the T=0.5 slice yields a signal
        assert len(signals) == 1
        assert signals[0].expiry_time == approx(0.5)

    # ------------------------------------------------------------------
    def test_skips_unsolvable_iv(self) -> None:
        """Quote with price below intrinsic (IV solver returns None) → skipped gracefully."""
        fs = _flat_fs(T=0.5)
        # Deep ITM call (K=80, spot=100) has intrinsic ≈ 21.98;
        # price = 1.0 is < intrinsic but > 0 (passes Pydantic) → IV solver returns None.
        surface = VolSurface(
            spot=100.0, risk_free=0.05, div_yield=0.0,
            slices=[ExpirySlice(expiry_time=0.5, quotes=[
                Quote(strike=80.0, option_type=OptionType.CALL, price=1.0),  # unsolvable
                Quote(strike=100.0, option_type=OptionType.CALL,
                      price=price_floats(100.0, 100.0, 0.5, 0.05, 0.0, 0.25, True)),
            ])],
        )
        signals = detect_mispricing(surface, fs, threshold=0.01, snapshot_date=_DUMMY_DATE)
        assert len(signals) == 1  # only the valid quote generates a signal
        assert signals[0].strike == approx(100.0)


# ===================================================================
#  P&L realisation tests
# ===================================================================

class TestRealizeTradePnL:
    """Tests for ``realize_trade_pnl``."""

    def _make_trade(
        self,
        option_type: OptionType = OptionType.CALL,
        strike: float = 100.0,
        side: int = 1,
        frozen_vol: float = 0.2,
        entry_price: float = 5.0,
        expiry_time: float = 0.5,
        r: float = 0.05,
        q: float = 0.0,
    ) -> Trade:
        expiry_date = _DUMMY_DATE + timedelta(days=round(expiry_time * 365))
        sg = MispricingSignal(
            strike=strike,
            expiry_time=expiry_time,
            expiry_date=expiry_date,
            option_type=option_type,
            market_iv=frozen_vol,
            model_iv=frozen_vol,
            mispricing=0.0,
            entry_price=entry_price,
            side=side,
        )
        return Trade(
            signal=sg,
            entry_date=_DUMMY_DATE,
            entry_spot=100.0,
            frozen_vol=frozen_vol,
            quantity=side,
            risk_free=r,
            div_yield=q,
        )

    # ------------------------------------------------------------------
    def test_long_call_flat_path(self) -> None:
        """Long call, flat path ending ATM → option_pnl = -entry_price, hedge ≈ 0."""
        trade = self._make_trade(entry_price=6.0)
        expiry_date = trade.signal.expiry_date
        path = _flat_path(_DUMMY_DATE, expiry_date, price=100.0)

        pnl = realize_trade_pnl(trade, path)
        # ATM at expiry → intrinsic = 0
        assert pnl.option_pnl == approx(-6.0, abs=1e-10)
        # Flat path → all ΔS = 0 → hedge_pnl = 0
        assert pnl.hedge_pnl == approx(0.0, abs=1e-10)
        assert pnl.realized_pnl == approx(pnl.option_pnl + pnl.hedge_pnl)
        assert pnl.hit is False
        assert pnl.hold_days == expiry_date.toordinal() - _DUMMY_DATE.toordinal()

    # ------------------------------------------------------------------
    def test_short_put_down_path(self) -> None:
        """Short put, path ending below strike → finite P&L, correct option_pnl."""
        trade = self._make_trade(
            option_type=OptionType.PUT, side=-1, entry_price=3.0,
        )
        expiry_date = trade.signal.expiry_date
        # Path drops from 100 to 90 monotonically
        path = _linear_path(_DUMMY_DATE, expiry_date, start_price=100.0, end_price=90.0)

        pnl = realize_trade_pnl(trade, path)
        # Intrinsic at S=90: max(100-90, 0) = 10
        expected_option = -1.0 * (10.0 - 3.0)  # qty * (intrinsic - entry)
        assert pnl.option_pnl == approx(expected_option, abs=1e-10)
        # Hedge P&L should be finite
        assert math.isfinite(pnl.hedge_pnl)
        assert pnl.realized_pnl == approx(pnl.option_pnl + pnl.hedge_pnl)
        assert pnl.expiry_spot == approx(90.0)

    # ------------------------------------------------------------------
    def test_at_expiry_uses_nearest_date(self) -> None:
        """Missing exact expiry_date in price_path → nearest-before used, no crash."""
        trade = self._make_trade(entry_price=5.0)
        expiry_date = trade.signal.expiry_date
        # Price path up to one day before expiry
        path: dict[date, float] = {}
        d = _DUMMY_DATE
        one_before = expiry_date - timedelta(days=1)
        while d <= one_before:
            path[d] = 100.0
            d += timedelta(days=1)

        pnl = realize_trade_pnl(trade, path)
        # Should use the last available date (one_before) as settlement
        assert pnl.expiry_spot == approx(100.0)
        assert math.isfinite(pnl.realized_pnl)

    # ------------------------------------------------------------------
    def test_empty_path_raises(self) -> None:
        """Empty price_path → ValueError."""
        trade = self._make_trade()
        with pytest.raises(ValueError, match="price_path has no date on or before entry_date"):
            realize_trade_pnl(trade, {})

    # ------------------------------------------------------------------
    def test_no_prior_date_raises(self) -> None:
        """Path with only dates after entry_date → ValueError (no prior date)."""
        trade = self._make_trade()
        # Path with only dates after entry_date
        later = _DUMMY_DATE + timedelta(days=1)
        path = {later: 100.0}
        with pytest.raises(ValueError, match="price_path has no date on or before entry_date"):
            realize_trade_pnl(trade, path)

    # ------------------------------------------------------------------
    def test_entry_date_missing_uses_prior_close(self) -> None:
        """Path lacks entry_date but has prior date → uses prior close as effective start."""
        trade = self._make_trade(entry_price=5.0)
        expiry_date = trade.signal.expiry_date
        one_day_before = _DUMMY_DATE - timedelta(days=1)
        # Path: starts one day before entry_date, goes to expiry
        path: dict[date, float] = {}
        d = one_day_before
        while d <= expiry_date:
            path[d] = 100.0
            d += timedelta(days=1)
        # entry_date is NOT in the path
        del path[_DUMMY_DATE]

        pnl = realize_trade_pnl(trade, path)
        # Should not raise; uses one_day_before as effective entry
        assert math.isfinite(pnl.realized_pnl)
        assert pnl.realized_pnl == approx(pnl.option_pnl + pnl.hedge_pnl)
        assert pnl.hold_days == expiry_date.toordinal() - _DUMMY_DATE.toordinal()

    # ------------------------------------------------------------------
    def test_hedge_sign_convention(self) -> None:
        """Long call on a monotonic up path → option_pnl > 0, hedge_pnl < 0."""
        trade = self._make_trade(entry_price=6.0)
        expiry_date = trade.signal.expiry_date
        path = _linear_path(_DUMMY_DATE, expiry_date, 100.0, 110.0)

        pnl = realize_trade_pnl(trade, path)
        # ITM at expiry
        assert pnl.option_pnl > 0.0
        # Hedge loses money on up-move (short delta shares when spot rises)
        assert pnl.hedge_pnl < 0.0
        # Both finite
        assert math.isfinite(pnl.realized_pnl)
        assert pnl.realized_pnl == approx(pnl.option_pnl + pnl.hedge_pnl)


# ===================================================================
#  Engine & metrics tests
# ===================================================================

class TestRunBacktest:
    """Tests for ``run_backtest``."""

    # ------------------------------------------------------------------
    def test_end_to_end_synthetic(self) -> None:
        """Full pipeline with 2 mispriced quotes and injected price path."""
        fs = _flat_fs(sigma=0.2, T=0.5)
        spot = 100.0
        r = 0.05
        q = 0.0

        # Overpriced call (IV=0.25) and underpriced put (IV=0.15)
        price_over = price_floats(spot, 100.0, 0.5, r, q, 0.25, True)
        price_under = price_floats(spot, 100.0, 0.5, r, q, 0.15, False)

        surface = VolSurface(
            spot=spot, risk_free=r, div_yield=q,
            slices=[ExpirySlice(expiry_time=0.5, quotes=[
                Quote(strike=100.0, option_type=OptionType.CALL, price=price_over),
                Quote(strike=100.0, option_type=OptionType.PUT, price=price_under),
            ])],
        )

        # Inject a deterministic flat price path
        expiry_date = _DUMMY_DATE + timedelta(days=round(0.5 * 365))
        path = _flat_path(_DUMMY_DATE, expiry_date, price=spot)

        called_with: list = []

        def _fake_fetch(sym: str, start: date, end: date) -> dict[date, float]:
            called_with.append((sym, start, end))
            return path

        result = run_backtest(
            surface=surface,
            fs=fs,
            symbol="SPY",
            snapshot_date=_DUMMY_DATE,
            threshold=0.01,
            fetch_prices=_fake_fetch,
        )

        assert result.n_trades == 2
        assert 0 <= result.hit_rate <= 1.0
        # Total P&L should match sum of individual PnLs
        expected_total = sum(p.realized_pnl for p in result.pnls)
        assert result.total_pnl == approx(expected_total, abs=1e-10)
        assert math.isfinite(result.sharpe)
        assert result.max_drawdown >= 0.0

    # ------------------------------------------------------------------
    def test_no_signals_returns_empty(self) -> None:
        """Clean surface with no mispricing → empty BacktestResult."""
        fs = _flat_fs(sigma=0.2, T=0.5)
        # Quote priced exactly at model IV
        price_fair = price_floats(100.0, 100.0, 0.5, 0.05, 0.0, 0.2, True)
        surface = VolSurface(
            spot=100.0, risk_free=0.05, div_yield=0.0,
            slices=[ExpirySlice(expiry_time=0.5, quotes=[
                Quote(strike=100.0, option_type=OptionType.CALL, price=price_fair),
            ])],
        )

        result = run_backtest(
            surface=surface, fs=fs, symbol="SPY",
            snapshot_date=_DUMMY_DATE, threshold=0.01,
            fetch_prices=lambda sym, s, e: {},
        )

        assert result.n_trades == 0
        assert result.trades == ()
        assert result.pnls == ()
        assert result.hit_rate == 0.0
        assert result.sharpe == 0.0
        assert result.total_pnl == 0.0
        assert result.max_drawdown == 0.0
        assert result.pnl_p5 == 0.0
        assert result.pnl_p50 == 0.0
        assert result.pnl_p95 == 0.0

    # ------------------------------------------------------------------
    def test_metrics_consistency(self) -> None:
        """With >= 2 trades: std = np.std(ddof=1), mean = total/n, drawdown >= 0."""
        fs = _flat_fs(sigma=0.2, T=0.5)
        spot = 100.0

        price1 = price_floats(spot, 100.0, 0.5, 0.05, 0.0, 0.25, True)   # overpriced call
        price2 = price_floats(spot, 100.0, 0.5, 0.05, 0.0, 0.25, False)  # overpriced put

        surface = VolSurface(
            spot=spot, risk_free=0.05, div_yield=0.0,
            slices=[ExpirySlice(expiry_time=0.5, quotes=[
                Quote(strike=100.0, option_type=OptionType.CALL, price=price1),
                Quote(strike=100.0, option_type=OptionType.PUT, price=price2),
            ])],
        )

        expiry_date = _DUMMY_DATE + timedelta(days=round(0.5 * 365))
        path = _flat_path(_DUMMY_DATE, expiry_date, price=spot)

        result = run_backtest(
            surface=surface, fs=fs, symbol="SPY",
            snapshot_date=_DUMMY_DATE, threshold=0.01,
            fetch_prices=lambda sym, s, e: path,
        )

        assert result.n_trades >= 2

        realized = np.array([p.realized_pnl for p in result.pnls])
        assert result.mean_pnl == approx(float(np.mean(realized)), abs=1e-10)
        assert result.std_pnl == approx(float(np.std(realized, ddof=1)), abs=1e-10)
        assert result.total_pnl == approx(float(np.sum(realized)), abs=1e-10)
        assert result.max_drawdown >= 0.0

    # ------------------------------------------------------------------
    def test_uses_injected_fetch_prices(self) -> None:
        """Injected callable is called with (symbol, snapshot_date, last_expiry_date)."""
        fs = _flat_fs(sigma=0.2, T=0.5)
        price_over = price_floats(100.0, 100.0, 0.5, 0.05, 0.0, 0.25, True)
        surface = VolSurface(
            spot=100.0, risk_free=0.05, div_yield=0.0,
            slices=[ExpirySlice(expiry_time=0.5, quotes=[
                Quote(strike=100.0, option_type=OptionType.CALL, price=price_over),
            ])],
        )

        spy_called: list = []

        def spy(sym: str, start: date, end: date) -> dict[date, float]:
            spy_called.append((sym, start, end))
            expiry_date = _DUMMY_DATE + timedelta(days=round(0.5 * 365))
            return _flat_path(_DUMMY_DATE, expiry_date)

        result = run_backtest(
            surface=surface, fs=fs, symbol="SPY",
            snapshot_date=_DUMMY_DATE, threshold=0.01,
            fetch_prices=spy,
        )

        assert len(spy_called) == 1
        sym, start, end = spy_called[0]
        assert sym == "SPY"
        assert start == _DUMMY_DATE
        # end should be the last expiry date
        assert end >= start


# ===================================================================
#  fetch_underlying_path (mocked)
# ===================================================================

class TestFetchUnderlyingPath:
    """Tests for ``fetch_underlying_path`` (mocked, no network)."""

    @patch("yfinance.Ticker")
    def test_mocked_path(self, mock_ticker_class) -> None:
        """Mock yfinance Ticker and verify returned dict structure."""
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        start = date(2030, 6, 15)
        end = date(2030, 6, 20)

        # Build a fake history DataFrame
        idx = pd.date_range("2030-06-15", "2030-06-20")
        df = pd.DataFrame(
            {"Close": [100.0, 101.0, 102.0, 99.0, 98.0, 97.0]},
            index=idx,
        )
        mock_ticker.history.return_value = df

        path = fetch_underlying_path("SPY", start, end)
        assert len(path) == 6
        assert path[date(2030, 6, 15)] == approx(100.0)
        assert path[date(2030, 6, 20)] == approx(97.0)
        # Verify history was called with the correct arguments
        call_kwargs = mock_ticker.history.call_args[1]
        assert call_kwargs["start"] == start - timedelta(days=5)
        # end should be bumped by 1 day (inclusive)
        assert call_kwargs["end"] == end + timedelta(days=1)

    @patch("yfinance.Ticker")
    def test_empty_data_raises(self, mock_ticker_class) -> None:
        """Empty DataFrame → ValueError."""
        import pandas as pd

        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        df = pd.DataFrame(columns=["Close"])
        mock_ticker.history.return_value = df

        with pytest.raises(ValueError, match="No price data"):
            fetch_underlying_path("SPY", date(2030, 1, 1), date(2030, 1, 5))
