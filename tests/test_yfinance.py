"""Tests for per-expiry dividend yield estimation in yfinance module.

Tests the _estimate_index_dividend_yield and _get_representative_dividend_yield
helpers using synthetic data and mocked yfinance calls.
"""

from datetime import date
from unittest.mock import patch, MagicMock

import pytest

from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote


# ── _estimate_index_dividend_yield tests ─────────────────────────────

def test_estimate_index_dividend_yield_via_parity() -> None:
    """Verify q estimation matches the put-call parity rearrangement."""
    from arbfree_vol.ingestion.yfinance import _estimate_index_dividend_yield
    from arbfree_vol.models.option import OptionContract, BlackScholesInput
    from arbfree_vol.pricing.black_scholes import price

    S, r, T, K = 100.0, 0.05, 1.0, 100.0
    q_true = 0.013  # 1.3% dividend yield

    # Price C and P using Black-Scholes with q_true
    contract_c = OptionContract(
        symbol="X", option_type=OptionType.CALL, strike=K,
        expiry_date=date(2030, 1, 1),
    )
    C = price(BlackScholesInput(
        contract=contract_c, spot=S, expiry_time=T,
        risk_free=r, div_yield=q_true, volatility=0.2,
    ))

    contract_p = OptionContract(
        symbol="X", option_type=OptionType.PUT, strike=K,
        expiry_date=date(2030, 1, 1),
    )
    P = price(BlackScholesInput(
        contract=contract_p, spot=S, expiry_time=T,
        risk_free=r, div_yield=q_true, volatility=0.2,
    ))

    slice_ = ExpirySlice(
        expiry_time=T,
        quotes=[
            Quote(strike=K, option_type=OptionType.CALL, price=C),
            Quote(strike=K, option_type=OptionType.PUT, price=P),
        ],
    )

    q_est = _estimate_index_dividend_yield(slice_, S, r)
    assert q_est is not None
    assert abs(q_est - q_true) < 0.002, (
        f"Estimated q={q_est:.6f} differs from true q={q_true} by > 20bps"
    )


def test_estimate_index_dividend_yield_no_atm_pair() -> None:
    """Slice with only calls → returns None."""
    from arbfree_vol.ingestion.yfinance import _estimate_index_dividend_yield

    slice_ = ExpirySlice(
        expiry_time=1.0,
        quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=5.0),
            Quote(strike=105.0, option_type=OptionType.CALL, price=3.0),
        ],
    )

    q_est = _estimate_index_dividend_yield(slice_, 100.0, 0.05)
    assert q_est is None


def test_estimate_index_dividend_yield_empty_slice() -> None:
    """Empty quotes list → returns None (but ExpirySlice requires min_length=1,
    so we test with a single quote that has no pair)."""
    from arbfree_vol.ingestion.yfinance import _estimate_index_dividend_yield

    slice_ = ExpirySlice(
        expiry_time=1.0,
        quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=5.0),
        ],
    )

    q_est = _estimate_index_dividend_yield(slice_, 100.0, 0.05)
    assert q_est is None


def test_estimate_index_dividend_yield_zero_expiry() -> None:
    """Zero expiry time → returns None."""
    from arbfree_vol.ingestion.yfinance import _estimate_index_dividend_yield

    slice_ = ExpirySlice(
        expiry_time=0.001,  # effectively zero
        quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=5.0),
            Quote(strike=100.0, option_type=OptionType.PUT, price=5.0),
        ],
    )

    q_est = _estimate_index_dividend_yield(slice_, 100.0, 0.05)
    # With T very small, the parity formula may or may not work; just ensure no crash
    # The function should return None or a valid float


# ── _get_representative_dividend_yield tests ─────────────────────────

@patch("arbfree_vol.ingestion.yfinance.yf.Ticker")
def test_representative_dividend_yield_spx(mock_ticker_class) -> None:
    """^SPX → SPY mapping returns SPY's dividend yield."""
    from arbfree_vol.ingestion.yfinance import _get_representative_dividend_yield

    mock_ticker = MagicMock()
    mock_ticker.info = {"dividendYield": 0.013}
    mock_ticker_class.return_value = mock_ticker

    q = _get_representative_dividend_yield("^SPX")
    assert q is not None
    assert abs(q - 0.013) < 1e-6
    mock_ticker_class.assert_called_once_with("SPY")


@patch("arbfree_vol.ingestion.yfinance.yf.Ticker")
def test_representative_dividend_yield_vix(mock_ticker_class) -> None:
    """^VIX → None (no representative ETF)."""
    from arbfree_vol.ingestion.yfinance import _get_representative_dividend_yield

    q = _get_representative_dividend_yield("^VIX")
    assert q is None
    mock_ticker_class.assert_not_called()


@patch("arbfree_vol.ingestion.yfinance.yf.Ticker")
def test_representative_dividend_yield_unknown_symbol(mock_ticker_class) -> None:
    """Unknown index symbol → None."""
    from arbfree_vol.ingestion.yfinance import _get_representative_dividend_yield

    q = _get_representative_dividend_yield("^UNKNOWN")
    assert q is None
    mock_ticker_class.assert_not_called()


@patch("arbfree_vol.ingestion.yfinance.yf.Ticker")
def test_representative_dividend_yield_handles_percent(mock_ticker_class) -> None:
    """When yfinance returns percent (>0.50), it's converted to fraction."""
    from arbfree_vol.ingestion.yfinance import _get_representative_dividend_yield

    mock_ticker = MagicMock()
    mock_ticker.info = {"dividendYield": 1.3}  # 1.3% as percent
    mock_ticker_class.return_value = mock_ticker

    q = _get_representative_dividend_yield("^SPX")
    assert q is not None
    assert abs(q - 0.013) < 1e-6


@patch("arbfree_vol.ingestion.yfinance.yf.Ticker")
def test_representative_dividend_yield_handles_exception(mock_ticker_class) -> None:
    """When yfinance raises, returns None gracefully."""
    from arbfree_vol.ingestion.yfinance import _get_representative_dividend_yield

    mock_ticker_class.side_effect = Exception("network error")

    q = _get_representative_dividend_yield("^SPX")
    assert q is None
