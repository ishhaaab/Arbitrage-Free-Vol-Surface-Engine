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
