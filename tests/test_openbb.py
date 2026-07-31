"""Tests for the OpenBB ingestion module."""

from unittest.mock import patch, MagicMock

import pytest

from arbfree_vol.ingestion import openbb as openbb_mod


class TestOpenBBImportGuard:
    """Test that the module handles missing openbb gracefully."""

    def test_fetch_chain_raises_import_error_when_openbb_missing(self):
        """fetch_chain raises ImportError with install instructions if openbb is absent."""
        with patch.dict("sys.modules", {"openbb": None}):
            with pytest.raises(ImportError, match="pip install openbb"):
                openbb_mod.fetch_chain("SPY")


class TestOpenBBColumnMapping:
    """Test the column normalisation helper."""

    def test_normalise_columns_renames_correctly(self):
        """_normalise_columns maps OpenBB names to yfinance-compatible names."""
        import pandas as pd
        df = pd.DataFrame({
            "open_interest": [100, 200],
            "last_trade_price": [1.5, 2.5],
            "implied_volatility": [0.2, 0.3],
            "strike": [100, 110],
            "bid": [1.0, 2.0],
            "ask": [2.0, 3.0],
            "volume": [10, 20],
        })
        result = openbb_mod._normalise_columns(df)
        assert "openInterest" in result.columns
        assert "lastPrice" in result.columns
        assert "impliedVolatility" in result.columns
        assert "open_interest" not in result.columns
        assert "last_trade_price" not in result.columns

    def test_normalise_columns_handles_nan(self):
        """_normalise_columns converts NaN values to 0."""
        import pandas as pd
        import math
        df = pd.DataFrame({
            "open_interest": [None, 200],
            "volume": [float("nan"), 20],
            "strike": [100, 110],
            "bid": [1.0, 2.0],
            "ask": [2.0, 3.0],
        })
        result = openbb_mod._normalise_columns(df)
        assert result.iloc[0]["openInterest"] == 0.0
        assert result.iloc[0]["volume"] == 0.0


class TestOpenBBRowToQuote:
    """Test the row-to-quote conversion."""

    def test_mid_price_used_when_bid_ask_available(self):
        """Mid price is computed from bid/ask."""
        row = {"strike": 100, "bid": 1.0, "ask": 2.0}
        from arbfree_vol.models.option import OptionType
        q = openbb_mod._row_to_quote(row, OptionType.CALL)
        assert q is not None
        assert q.price == pytest.approx(1.5)
        assert q.strike == 100.0

    def test_fallback_to_last_trade_price(self):
        """Falls back to last_trade_price when bid/ask missing."""
        row = {"strike": 100, "bid": None, "ask": None, "last_trade_price": 3.0}
        from arbfree_vol.models.option import OptionType
        q = openbb_mod._row_to_quote(row, OptionType.PUT)
        assert q is not None
        assert q.price == pytest.approx(3.0)

    def test_returns_none_when_no_price(self):
        """Returns None when no valid price source exists."""
        row = {"strike": 100, "bid": None, "ask": None, "last_trade_price": None}
        from arbfree_vol.models.option import OptionType
        q = openbb_mod._row_to_quote(row, OptionType.CALL)
        assert q is None


class TestSafeConversions:
    """Test the _safe_int and _safe_float helpers."""

    def test_safe_int_with_none(self):
        assert openbb_mod._safe_int(None) == 0
        assert openbb_mod._safe_int(None, default=5) == 5

    def test_safe_int_with_nan(self):
        assert openbb_mod._safe_int(float("nan")) == 0

    def test_safe_int_with_valid(self):
        assert openbb_mod._safe_int(42) == 42

    def test_safe_float_with_none(self):
        assert openbb_mod._safe_float(None) == 0.0

    def test_safe_float_with_nan(self):
        assert openbb_mod._safe_float(float("nan")) == 0.0

    def test_safe_float_with_valid(self):
        assert openbb_mod._safe_float(3.14) == pytest.approx(3.14)


class TestOpenBBIndexDividendYield:
    """Test per-expiry dividend yield estimation for index symbols."""

    def test_estimate_via_parity(self) -> None:
        """Verify q estimation matches the put-call parity rearrangement."""
        from datetime import date as date_cls
        from arbfree_vol.models.option import OptionContract, BlackScholesInput
        from arbfree_vol.models.option import OptionType as OT
        from arbfree_vol.pricing.black_scholes import price
        from arbfree_vol.models.surface import ExpirySlice, Quote

        S, r, T, K = 100.0, 0.05, 1.0, 100.0
        q_true = 0.013

        contract_c = OptionContract(
            symbol="X", option_type=OT.CALL, strike=K,
            expiry_date=date_cls(2030, 1, 1),
        )
        C = price(BlackScholesInput(
            contract=contract_c, spot=S, expiry_time=T,
            risk_free=r, div_yield=q_true, volatility=0.2,
        ))

        contract_p = OptionContract(
            symbol="X", option_type=OT.PUT, strike=K,
            expiry_date=date_cls(2030, 1, 1),
        )
        P = price(BlackScholesInput(
            contract=contract_p, spot=S, expiry_time=T,
            risk_free=r, div_yield=q_true, volatility=0.2,
        ))

        slice_ = ExpirySlice(
            expiry_time=T,
            quotes=[
                Quote(strike=K, option_type=OT.CALL, price=C),
                Quote(strike=K, option_type=OT.PUT, price=P),
            ],
        )

        q_est = openbb_mod._estimate_index_dividend_yield(slice_, S, r)
        assert q_est is not None
        assert abs(q_est - q_true) < 0.002

    def test_no_atm_pair(self) -> None:
        """Slice with only calls → returns None."""
        from arbfree_vol.models.option import OptionType as OT
        from arbfree_vol.models.surface import ExpirySlice, Quote

        slice_ = ExpirySlice(
            expiry_time=1.0,
            quotes=[
                Quote(strike=100.0, option_type=OT.CALL, price=5.0),
                Quote(strike=105.0, option_type=OT.CALL, price=3.0),
            ],
        )

        q_est = openbb_mod._estimate_index_dividend_yield(slice_, 100.0, 0.05)
        assert q_est is None

    @patch("yfinance.Ticker")
    def test_representative_spx(self, mock_ticker_class) -> None:
        """^SPX → SPY mapping returns SPY's dividend yield."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"dividendYield": 0.013}
        mock_ticker_class.return_value = mock_ticker

        q = openbb_mod._get_representative_dividend_yield("^SPX")
        assert q is not None
        assert abs(q - 0.013) < 1e-6
        mock_ticker_class.assert_called_once_with("SPY")

    def test_representative_vix(self) -> None:
        """^VIX → None (no representative ETF)."""
        q = openbb_mod._get_representative_dividend_yield("^VIX")
        assert q is None
