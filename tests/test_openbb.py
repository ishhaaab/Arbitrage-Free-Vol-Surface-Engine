"""Tests for the OpenBB ingestion module."""

import logging
import sys
import types
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from arbfree_vol.ingestion import openbb as openbb_mod
from arbfree_vol.ingestion._common import row_to_quote
from arbfree_vol.ingestion._index_rates import (
    _estimate_index_dividend_yield,
    _get_representative_dividend_yield,
)
from arbfree_vol.models.surface import VolSurface


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

    def test_normalise_columns_preserves_missing(self):
        """_normalise_columns keeps missing values as NaN (not 0.0) so the
        quality filter can distinguish a missing value from a genuinely
        observed zero (the missing-OI-as-0 bug class)."""
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
        assert math.isnan(result.iloc[0]["openInterest"])
        assert math.isnan(result.iloc[0]["volume"])


class TestOpenBBRowToQuote:
    """Test the row-to-quote conversion."""

    def test_mid_price_used_when_bid_ask_available(self):
        """Mid price is computed from bid/ask."""
        row = {"strike": 100, "bid": 1.0, "ask": 2.0}
        from arbfree_vol.models.option import OptionType
        q = row_to_quote(row, OptionType.CALL)
        assert q is not None
        assert q.price == pytest.approx(1.5)
        assert q.strike == 100.0

    def test_fallback_to_last_price(self):
        """Falls back to the normalized lastPrice when bid/ask missing."""
        row = {"strike": 100, "bid": None, "ask": None, "lastPrice": 3.0}
        from arbfree_vol.models.option import OptionType
        q = row_to_quote(row, OptionType.PUT)
        assert q is not None
        assert q.price == pytest.approx(3.0)

    def test_normalized_row_missing_bid_ask_falls_back_to_last_price(self):
        """A REAL normalized OpenBB row (missing/NaN bid AND ask but a
        valid lastPrice) must produce a quote via the last-price fallback,
        not be silently dropped.

        End-to-end regression for the pre-fix bug: ``_normalise_columns``
        renames ``last_trade_price`` → ``lastPrice``, but
        ``_row_to_quote`` read the pre-normalization key, so a normalized
        row with missing bid/ask fell through the mid-price branch and
        returned None.  Drives the same normalise-then-convert path the
        fetch pipeline uses for row conversion."""
        import pandas as pd
        from arbfree_vol.models.option import OptionType

        raw = pd.DataFrame({
            "strike": [100.0],
            "bid": [float("nan")],
            "ask": [None],
            "last_trade_price": [3.0],
        })
        normalized = openbb_mod._normalise_columns(raw)
        q = row_to_quote(normalized.iloc[0], OptionType.PUT)
        assert q is not None
        assert q.price == pytest.approx(3.0)

    def test_returns_none_when_no_price(self):
        """Returns None when no valid price source exists."""
        row = {"strike": 100, "bid": None, "ask": None, "lastPrice": None}
        from arbfree_vol.models.option import OptionType
        q = row_to_quote(row, OptionType.CALL)
        assert q is None


class TestOpenBBExpirationParsing:
    """Test the expiration → date-string normalization used by the
    fetch_chain sort/parse path."""

    def test_expiration_datetime_parses_to_date_string(self):
        """A datetime/pandas.Timestamp expiration must be normalized to a
        date-only string so ``date.fromisoformat`` (used by fetch_chain's
        sort/parse path) cannot be broken by a time component."""
        from datetime import datetime as dt_cls
        import pandas as pd

        df = pd.DataFrame({
            "expiration": [
                pd.Timestamp("2026-08-11 09:30:00"),
                dt_cls(2026, 9, 15, 16, 0, 0),
                "2026-10-01T00:00:00",
            ],
        })
        parsed = df["expiration"].apply(openbb_mod._expiry_to_date_str)
        assert list(parsed) == ["2026-08-11", "2026-09-15", "2026-10-01"]
        # The sort/parse path in fetch_chain must accept every value.
        for s in parsed:
            date.fromisoformat(s)


class TestSafeConversions:
    """Test the _safe_float helper."""

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

        q_est = _estimate_index_dividend_yield(slice_, S, r)
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

        q_est = _estimate_index_dividend_yield(slice_, 100.0, 0.05)
        assert q_est is None

    @patch("yfinance.Ticker")
    def test_representative_spx(self, mock_ticker_class) -> None:
        """^SPX → SPY mapping returns SPY's dividend yield."""
        mock_ticker = MagicMock()
        mock_ticker.info = {"dividendYield": 0.013}
        mock_ticker_class.return_value = mock_ticker

        q = _get_representative_dividend_yield("^SPX")
        assert q is not None
        assert abs(q - 0.013) < 1e-6
        mock_ticker_class.assert_called_once_with("SPY")

    def test_representative_vix(self) -> None:
        """^VIX → None (no representative ETF)."""
        q = _get_representative_dividend_yield("^VIX")
        assert q is None


class _FakeTicker:
    """yfinance.Ticker stand-in whose ``.info`` is canned per symbol.

    ``^IRX`` yields the 13-week T-bill rate (r = 5.0 / 100 = 0.05),
    ``SPY`` yields a real dividend yield (q = 0.011), and index symbols
    like ``^SPX`` yield an empty info dict so the parity path runs.
    """

    _INFO = {
        "^IRX": {"regularMarketPrice": 5.0},
        "SPY": {"dividendYield": 0.011},
        "^SPX": {},
    }

    def __init__(self, symbol: str):
        self._symbol = symbol

    @property
    def info(self) -> dict:
        return dict(self._INFO.get(self._symbol, {}))


def _chains_df(spot: float = 100.0) -> pd.DataFrame:
    """Build a canonical synthetic OpenBB chain DataFrame.

    Two expiries (~30 and ~60 days out), each with calls + puts at three
    strikes around ``spot``.  Quotes are non-crossed with positive bids
    and a mid that carries time value over intrinsic, so they pass both
    the data-quality filter and the cleaning layer.  Exact pricing does
    not matter — only that the rows are realistic enough to survive the
    pipeline end-to-end.
    """
    rows = []
    for days in (30, 60):
        expiration = (date.today() + timedelta(days=days)).isoformat()
        for strike_frac in (0.9, 1.0, 1.1):
            strike = spot * strike_frac
            for otype, intrinsic in (
                ("call", max(0.0, spot - strike)),
                ("put", max(0.0, strike - spot)),
            ):
                mid = intrinsic + 2.5
                rows.append({
                    "strike": strike,
                    "option_type": otype,
                    "expiration": expiration,
                    "bid": round(mid * 0.95, 2),
                    "ask": round(mid * 1.05, 2),
                    "last_trade_price": round(mid, 2),
                    "open_interest": 100,
                    "volume": 10,
                    "underlying_price": spot,
                })
    return pd.DataFrame(rows)


def _fake_openbb(monkeypatch, chains_df: pd.DataFrame) -> MagicMock:
    """Inject a fake ``openbb`` module and a fake ``yfinance.Ticker``.

    ``obb.derivatives.options.chains(...)`` returns an object whose
    ``.to_df()`` yields ``chains_df``, so the real ``fetch_chain`` main
    path runs end-to-end without the ``openbb`` package installed.
    ``yfinance.Ticker`` is patched so the ^IRX / dividend-yield lookups
    hit canned info instead of the network.  Returns the fake ``obb`` so
    callers can configure additional endpoints (e.g. equity quotes).
    """
    fake_openbb = types.ModuleType("openbb")
    fake_obb = MagicMock()
    fake_obb.derivatives.options.chains.return_value.to_df.return_value = chains_df
    fake_openbb.obb = fake_obb
    monkeypatch.setitem(sys.modules, "openbb", fake_openbb)
    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    return fake_obb


class TestOpenBBFetchChainMainPath:
    """End-to-end fetch_chain main path against a fake openbb module."""

    def test_fetch_chain_main_path_builds_surface(self, monkeypatch) -> None:
        """SPY with the quality filter ON builds a full VolSurface."""
        _fake_openbb(monkeypatch, _chains_df())

        surface, rejected, quality_drops = openbb_mod.fetch_chain("SPY")

        assert isinstance(surface, VolSurface)
        assert surface.spot == pytest.approx(100.0)
        assert len(surface.slices) >= 1
        assert all(len(sl.quotes) > 0 for sl in surface.slices)
        # r comes from the fake ^IRX info (5.0 / 100), q from SPY's info
        assert surface.risk_free == pytest.approx(0.05)
        assert surface.div_yield == pytest.approx(0.011)
        assert isinstance(rejected, list)
        assert isinstance(quality_drops, list)

    def test_fetch_chain_observed_zero_dividend_yield_is_preserved(
        self, monkeypatch, caplog,
    ) -> None:
        """An OBSERVED zero dividend yield (SPY info carries
        ``dividendYield=0.0``) is preserved as q=0.0 and the warning says
        it was observed as zero — NOT substituted as missing (the pre-fix
        bug conflated q==0 with a missing field)."""
        fake_openbb = types.ModuleType("openbb")
        fake_obb = MagicMock()
        fake_obb.derivatives.options.chains.return_value.to_df.return_value = (
            _chains_df()
        )
        fake_openbb.obb = fake_obb
        monkeypatch.setitem(sys.modules, "openbb", fake_openbb)

        class _ZeroYieldTicker(_FakeTicker):
            _INFO = {
                "^IRX": {"regularMarketPrice": 5.0},
                "SPY": {"dividendYield": 0.0},  # observed zero, present
                "^SPX": {},
            }

        monkeypatch.setattr("yfinance.Ticker", _ZeroYieldTicker)

        with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.openbb"):
            surface, _, _ = openbb_mod.fetch_chain("SPY")

        assert surface.div_yield == 0.0
        assert "observed as zero" in caplog.text, (
            f"expected the observed-zero warning, got: {caplog.text}"
        )
        assert "substituting q=0.0" not in caplog.text, (
            f"an observed zero must NOT be logged as a substitution, got: "
            f"{caplog.text}"
        )

    def test_fetch_chain_index_representative_zero_yield_preserved(
        self, monkeypatch, caplog,
    ) -> None:
        """Index path: all slices fail parity and the representative
        ETF's dividend yield is PRESENT as 0.0.  The surface q must be
        0.0 AS OBSERVED (not treated as missing), the representative
        helper must log "observed as zero", and fetch_chain must NOT
        claim the representative yield was unavailable or hardcoded.

        Regression (pre-fix): ``_get_representative_dividend_yield``
        required ``q > 0``, so a present-zero representative yield
        returned None and fetch_chain logged "no representative ETF
        yield available; surface q hardcoded to 0.0" — conflating an
        observed zero with missing data (commit 5bf429a aligned the
        primary paths; this pins the same semantics on the OpenBB
        representative fallback)."""
        fake_openbb = types.ModuleType("openbb")
        fake_obb = MagicMock()
        fake_obb.derivatives.options.chains.return_value.to_df.return_value = (
            _chains_df(spot=100.0)
        )
        fake_openbb.obb = fake_obb
        monkeypatch.setitem(sys.modules, "openbb", fake_openbb)

        class _ZeroRepTicker(_FakeTicker):
            _INFO = {
                "^IRX": {"regularMarketPrice": 5.0},
                "^SPX": {},
                "SPY": {"dividendYield": 0.0},  # representative, present zero
            }

        monkeypatch.setattr("yfinance.Ticker", _ZeroRepTicker)
        # Parity estimation fails for all slices so the representative
        # fallback path runs with the REAL helper (not monkeypatched).
        monkeypatch.setattr(
            "arbfree_vol.ingestion._index_rates._estimate_index_dividend_yield",
            lambda sl, spot, r: None,
        )

        with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.openbb"):
            surface, _, _ = openbb_mod.fetch_chain("^SPX")

        assert surface.div_yield == 0.0
        # The helper's observed-zero provenance is logged with its exact
        # wording (unique to the representative path).
        assert "representative ETF SPY has dividendYield present as 0.0" in caplog.text, (
            f"expected the representative observed-zero warning, got: "
            f"{caplog.text}"
        )
        assert "no representative ETF yield available" not in caplog.text, (
            f"a present-zero representative yield must not be logged as "
            f"unavailable, got: {caplog.text}"
        )
        assert "hardcoded to 0.0" not in caplog.text, (
            f"a present-zero representative yield must not be logged as a "
            f"hardcoded substitution, got: {caplog.text}"
        )

    def test_fetch_chain_disable_quality_filter(self, monkeypatch) -> None:
        """disable_quality_filter=True skips the filter entirely."""
        df = _chains_df()
        _fake_openbb(monkeypatch, df)

        surface, rejected, quality_drops = openbb_mod.fetch_chain(
            "SPY", disable_quality_filter=True
        )

        assert len(quality_drops) == 0
        assert len(surface.slices) >= 1
        # No filter means no drops, so at least as many quotes as filtered
        quotes_disabled = sum(len(sl.quotes) for sl in surface.slices)
        surface_filtered, _, _ = openbb_mod.fetch_chain("SPY")
        quotes_filtered = sum(len(sl.quotes) for sl in surface_filtered.slices)
        assert quotes_disabled >= quotes_filtered

    def test_fetch_chain_index_symbol_uses_parity_q(self, monkeypatch) -> None:
        """^SPX with empty info exercises the per-slice parity q path."""
        _fake_openbb(monkeypatch, _chains_df())

        surface, rejected, quality_drops = openbb_mod.fetch_chain("^SPX")

        assert isinstance(surface, VolSurface)
        assert any(sl.div_yield is not None for sl in surface.slices) \
            or surface.div_yield != 0.0

    def test_fetch_chain_index_placeholder_zero_not_logged_as_observed(
        self, monkeypatch, caplog,
    ) -> None:
        """Index symbols set a q=0.0 PLACEHOLDER before per-expiry
        put-call parity estimation — not an observed dividend yield.  The
        pre-loop log must say so (index-default path) and must NOT claim
        the yield was "observed as zero".

        Regression (pre-fix): the placeholder ``q = 0.0`` triggered the
        ``q == 0.0`` observed-zero branch, so EVERY index symbol logged
        "Dividend yield ... observed as zero (dividendYield present as
        0.0 in ticker info)" even though the value never came from the
        ticker's info (the yfinance path scopes that branch to non-index
        tickers; the OpenBB path did not)."""
        _fake_openbb(monkeypatch, _chains_df())

        with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.openbb"):
            openbb_mod.fetch_chain("^SPX")

        assert "observed as zero" not in caplog.text, (
            f"an index placeholder q=0.0 must not be logged as an "
            f"observed zero, got: {caplog.text}"
        )
        assert "index default" in caplog.text, (
            f"expected the index-default message for the q=0.0 "
            f"placeholder, got: {caplog.text}"
        )

    def test_fetch_chain_missing_expiration_raises(self, monkeypatch) -> None:
        """A chain without an 'expiration' column raises ValueError."""
        df = _chains_df().drop(columns=["expiration"])
        _fake_openbb(monkeypatch, df)

        with pytest.raises(ValueError):
            openbb_mod.fetch_chain("SPY")

    def test_fetch_chain_empty_df_raises(self, monkeypatch) -> None:
        """An empty chain DataFrame raises ValueError."""
        _fake_openbb(monkeypatch, pd.DataFrame())

        with pytest.raises(ValueError):
            openbb_mod.fetch_chain("SPY")

    def test_fetch_chain_spot_fallback_via_equity_quote(self, monkeypatch) -> None:
        """Missing underlying_price falls back to obb equity price quote."""
        df = _chains_df().drop(columns=["underlying_price"])
        fake_obb = _fake_openbb(monkeypatch, df)
        fake_obb.equity.price.quote.return_value.to_df.return_value = pd.DataFrame(
            {"last_price": [100.0]}
        )

        surface, rejected, quality_drops = openbb_mod.fetch_chain("SPY")

        assert isinstance(surface, VolSurface)
        assert surface.spot == pytest.approx(100.0)

    def test_fetch_chain_no_slices_raises(self, monkeypatch) -> None:
        """A chain whose only expiry is today produces no slices."""
        today = date.today().isoformat()
        rows = []
        for strike_frac in (0.9, 1.0, 1.1):
            strike = 100.0 * strike_frac
            for otype, intrinsic in (
                ("call", max(0.0, 100.0 - strike)),
                ("put", max(0.0, strike - 100.0)),
            ):
                mid = intrinsic + 2.5
                rows.append({
                    "strike": strike,
                    "option_type": otype,
                    "expiration": today,
                    "bid": round(mid * 0.95, 2),
                    "ask": round(mid * 1.05, 2),
                    "last_trade_price": round(mid, 2),
                    "open_interest": 100,
                    "volume": 10,
                    "underlying_price": 100.0,
                })
        _fake_openbb(monkeypatch, pd.DataFrame(rows))

        with pytest.raises(ValueError, match="No valid slices"):
            openbb_mod.fetch_chain("SPY")
