"""Targeted regression tests pinning two committed ingestion/quality fixes.

FIX A — "missing-vs-zero DropRecord" (commit a7cb71a):
    ``filter_option_chain`` / ``_is_missing`` distinguish a MISSING value
    (None / NaN / pd.NA) from a genuinely observed zero in
    ``openInterest``, ``bid`` and ``ask``.  A missing OI drops as
    ``OI=missing<10`` with ``missing_fields=("open_interest",)``; an
    observed ``OI=0`` drops as ``OI=0<10`` with no missing fields.  A
    one-sided quote (exactly one of bid/ask missing) drops as
    ``spread=missing (missing: <side>)`` instead of passing with a mid
    fabricated from the available side.  The OpenBB path preserves
    missingness as NaN in ``_normalise_columns`` so the shared filter
    sees the same missing values the yfinance path does.

FIX B — "r/q fallback logging" (commit b835b60):
    the yfinance and openbb ingestion paths log an explicit WARNING when
    the ^IRX risk-free-rate fetch fails ("... substituting r=0.05") and
    when ``dividendYield`` is missing ("... substituting q=0.0").  No
    silent fallbacks.

Every test here is written to FAIL if the corresponding fix is reverted
(verified by temporarily reversing each fix and re-running the file).
"""

import logging
import math
import sys
import types
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from arbfree_vol.data.quality import filter_option_chain
from arbfree_vol.ingestion import openbb as openbb_mod

from tests.chain_helpers import _make_chain_df


class TestFixA_MissingVsZeroOpenInterest:
    """FIX A: open interest — absent/NaN is MISSING, 0 is observed."""

    def test_genuinely_zero_oi_is_not_missing(self):
        """An observed OI=0 is dropped for the zero reason and carries no
        missing_fields — never byte-identical to a missing value."""
        df = _make_chain_df(
            strikes=[100.0],
            oi=[0],
            volume=[10],
            bid=[9.0],
            ask=[11.0],
        )
        filtered, drops = filter_option_chain(df, "2026-08-15")
        assert len(drops) == 1
        assert len(filtered) == 0
        assert drops[0].strike == 100.0
        assert "OI=0<10" in drops[0].reason
        assert "missing" not in drops[0].reason
        assert drops[0].missing_fields == ()
        assert drops[0].open_interest == 0

    def test_absent_oi_is_missing(self):
        """A None openInterest is MISSING: dropped with OI=missing and
        missing_fields=('open_interest',)."""
        df = _make_chain_df(
            strikes=[100.0],
            oi=[None],
            volume=[10],
            bid=[9.0],
            ask=[11.0],
        )
        _, drops = filter_option_chain(df, "2026-08-15")
        assert len(drops) == 1
        assert "OI=missing<10" in drops[0].reason
        assert drops[0].missing_fields == ("open_interest",)

    def test_nan_oi_is_missing(self):
        """A NaN openInterest is MISSING, not zero."""
        df = _make_chain_df(
            strikes=[100.0],
            oi=[float("nan")],
            volume=[10],
            bid=[9.0],
            ask=[11.0],
        )
        _, drops = filter_option_chain(df, "2026-08-15")
        assert len(drops) == 1
        assert "OI=missing<10" in drops[0].reason
        assert drops[0].missing_fields == ("open_interest",)

    def test_zero_and_missing_oi_are_distinguishable(self):
        """Zero, None and NaN OI produce distinct DropRecords side by side."""
        df = _make_chain_df(
            strikes=[100.0, 110.0, 120.0],
            oi=[0, None, float("nan")],
            volume=[10, 10, 10],
            bid=[9.0, 9.0, 9.0],
            ask=[11.0, 11.0, 11.0],
        )
        _, drops = filter_option_chain(df, "2026-08-15")
        by_strike = {d.strike: d for d in drops}
        assert len(by_strike) == 3
        assert "OI=0<10" in by_strike[100.0].reason
        assert by_strike[100.0].missing_fields == ()
        assert "OI=missing<10" in by_strike[110.0].reason
        assert by_strike[110.0].missing_fields == ("open_interest",)
        assert "OI=missing<10" in by_strike[120.0].reason
        assert by_strike[120.0].missing_fields == ("open_interest",)

    def test_is_missing_helper_distinguishes_zero(self):
        """The _is_missing helper itself: 0/0.0 are NOT missing, while
        None / NaN / pd.NA / pd.NaT ARE."""
        from arbfree_vol.data.quality import _is_missing

        assert not _is_missing(0)
        assert not _is_missing(0.0)
        assert not _is_missing(42)
        assert _is_missing(None)
        assert _is_missing(float("nan"))
        assert _is_missing(pd.NA)
        assert _is_missing(pd.NaT)


class TestFixA_MissingVsZeroQuotes:
    """FIX A: bid/ask sides — a one-sided quote is dropped naming the
    missing side; a genuinely zero side is a spread violation."""

    def test_missing_bid_is_flagged(self):
        """Missing bid + present ask → drop naming 'bid', not a
        fabricated spread number."""
        df = _make_chain_df(
            strikes=[100.0],
            oi=[100],
            volume=[10],
            bid=[float("nan")],
            ask=[10.0],
        )
        filtered, drops = filter_option_chain(df, "2026-08-15")
        assert len(drops) == 1
        assert len(filtered) == 0
        assert "spread=missing (missing: bid)" in drops[0].reason
        assert drops[0].missing_fields == ("bid",)

    def test_missing_ask_is_flagged_not_passed_with_fabricated_mid(self):
        """Regression: pre-fix, a missing ask passed silently with
        mid=bid/2 (negative spread never tripped the threshold).  The
        committed behaviour drops it naming 'ask'."""
        df = _make_chain_df(
            strikes=[100.0],
            oi=[100],
            volume=[10],
            bid=[5.0],
            ask=[float("nan")],
        )
        filtered, drops = filter_option_chain(df, "2026-08-15")
        assert len(drops) == 1
        assert len(filtered) == 0
        assert "spread=missing (missing: ask)" in drops[0].reason
        assert drops[0].missing_fields == ("ask",)

    def test_zero_bid_is_spread_violation_not_missing(self):
        """An observed bid=0 with a live ask is a 200% spread violation,
        NOT a 'missing: bid' flag."""
        df = _make_chain_df(
            strikes=[100.0],
            oi=[100],
            volume=[10],
            bid=[0.0],
            ask=[10.0],
        )
        _, drops = filter_option_chain(df, "2026-08-15")
        assert len(drops) == 1
        assert "spread=200.0%>50.0%" in drops[0].reason
        assert "missing" not in drops[0].reason
        assert drops[0].missing_fields == ()

    def test_zero_ask_is_not_flagged_missing(self):
        """An observed ask=0 with a live bid yields a NEGATIVE spread
        (never > threshold) so the row passes through — but it must NOT
        be flagged as a missing ask.  Treating a genuine zero as absent
        (the FIX A regression) would turn this row into a
        'spread=missing (missing: ask)' drop."""
        df = _make_chain_df(
            strikes=[100.0],
            oi=[100],
            volume=[10],
            bid=[10.0],
            ask=[0.0],
        )
        filtered, drops = filter_option_chain(df, "2026-08-15")
        assert len(filtered) == 1
        assert len(drops) == 0

    def test_both_sides_missing_dropped(self):
        """Both bid and ask missing → the quote has NO market data at
        all and is dropped naming both sides.  Regression: pre-fix,
        mid=0 skipped the spread branch so the row passed the filter
        with only OI as a criterion, and the quote later entered the
        pipeline priced at lastPrice (the N1 no-quote path)."""
        df = _make_chain_df(
            strikes=[100.0],
            oi=[100],
            volume=[10],
            bid=[float("nan")],
            ask=[float("nan")],
        )
        filtered, drops = filter_option_chain(df, "2026-08-15")
        assert len(filtered) == 0
        assert len(drops) == 1
        assert "spread=missing (missing: bid, ask)" in drops[0].reason
        assert drops[0].missing_fields == ("bid", "ask")

    def test_missing_volume_recorded_not_reason(self):
        """Missing volume is recorded in missing_fields but never becomes
        a filter reason (volume is not a criterion)."""
        df = _make_chain_df(
            strikes=[100.0],
            oi=[3],
            volume=[None],
            bid=[9.0],
            ask=[11.0],
        )
        _, drops = filter_option_chain(df, "2026-08-15")
        assert len(drops) == 1
        assert "volume" in drops[0].missing_fields
        assert "vol" not in drops[0].reason


class TestFixA_OpenBBPreservesMissingness:
    """FIX A (openbb half): _normalise_columns preserves missing values as
    NaN instead of coercing them to 0.0, so the shared filter sees the
    same missingness the yfinance path does."""

    def test_normalise_columns_keeps_missing_as_nan(self):
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

    def test_normalise_columns_preserves_real_zeros(self):
        """Observed zeros must stay zeros through normalisation — a
        missing value must never be conflated with a real zero."""
        df = pd.DataFrame({
            "open_interest": [0, 200],
            "volume": [0, 20],
            "strike": [100, 110],
            "bid": [0.0, 2.0],
            "ask": [2.0, 3.0],
        })
        result = openbb_mod._normalise_columns(df)
        assert result.iloc[0]["openInterest"] == 0.0
        assert result.iloc[0]["volume"] == 0.0


class TestFixA_IngestionPaths:
    """FIX A end-to-end: the real ingestion paths surface the
    missing-vs-zero semantics through their quality_drops audit trail."""

    @patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
    @patch("arbfree_vol.ingestion.yahoo.date")
    def test_yfinance_fetch_chain_reports_missing_and_zero_oi(
        self, mock_date_class, mock_ticker_class
    ) -> None:
        """yfinance fetch_chain: a NaN-OI strike drops as OI=missing,
        an observed OI=0 strike drops as OI=0<10 — never the same."""
        from datetime import date as real_date
        from arbfree_vol.ingestion.yahoo import fetch_chain

        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.info = {"regularMarketPrice": 450.0, "dividendYield": 0.005}
        mock_ticker.options = ["2030-08-15", "2030-09-15"]

        today = real_date(2030, 7, 15)
        mock_date_class.today.return_value = today
        mock_date_class.fromisoformat.side_effect = real_date.fromisoformat

        mock_irx = MagicMock()
        mock_irx.info = {"regularMarketPrice": 4.85}
        mock_ticker_class.side_effect = lambda s: (
            mock_irx if s == "^IRX" else mock_ticker
        )

        strikes = [430, 440, 450, 460, 470]
        cols = {
            "strike": strikes,
            "lastPrice": [25, 20, 15, 10, 5],
            "bid": [24, 19, 14, 9, 4],
            "ask": [26, 21, 16, 11, 6],
            "volume": [100, 100, 100, 100, 100],
            # 460: genuinely zero OI; 470: missing OI
            "openInterest": [500, 500, 500, 0, float("nan")],
        }
        mock_chain = MagicMock()
        mock_chain.calls = pd.DataFrame(cols | {"contractSymbol": ["c1", "c2", "c3", "c4", "c5"]})
        mock_chain.puts = pd.DataFrame(cols | {"contractSymbol": ["p1", "p2", "p3", "p4", "p5"]})
        mock_ticker.option_chain.return_value = mock_chain

        _, _, quality_drops = fetch_chain("SPY", max_expiries=2)

        by_strike = {d.strike: d for d in quality_drops}
        assert 470.0 in by_strike
        assert "OI=missing<10" in by_strike[470.0].reason
        assert by_strike[470.0].missing_fields == ("open_interest",)
        assert 460.0 in by_strike
        assert "OI=0<10" in by_strike[460.0].reason
        assert by_strike[460.0].missing_fields == ()

    def test_openbb_fetch_chain_reports_missing_and_zero_oi(self, monkeypatch) -> None:
        """openbb fetch_chain: a missing-OI row drops as OI=missing with
        missing_fields, an observed OI=0 row drops as OI=0<10."""
        rows = self._openbb_chains_df().to_dict("records")
        expiration = (date.today() + timedelta(days=30)).isoformat()
        rows.append({
            "strike": 100.0,
            "option_type": "call",
            "expiration": expiration,
            "bid": 2.0,
            "ask": 3.0,
            "last_trade_price": 2.5,
            "open_interest": None,  # missing, not zero
            "volume": 10,
            "underlying_price": 100.0,
        })
        rows.append({
            "strike": 100.0,
            "option_type": "put",
            "expiration": expiration,
            "bid": 2.0,
            "ask": 3.0,
            "last_trade_price": 2.5,
            "open_interest": 0,  # genuinely observed zero
            "volume": 10,
            "underlying_price": 100.0,
        })
        self._fake_openbb_env(
            monkeypatch,
            pd.DataFrame(rows),
            {"^IRX": {"regularMarketPrice": 5.0}, "SPY": {"dividendYield": 0.011}},
        )

        _, _, quality_drops = openbb_mod.fetch_chain("SPY")

        assert any("OI=missing<10" in d.reason for d in quality_drops)
        missing_drops = [d for d in quality_drops if "OI=missing<10" in d.reason]
        assert missing_drops
        assert all(d.missing_fields == ("open_interest",) for d in missing_drops)
        zero_drops = [d for d in quality_drops if "OI=0<10" in d.reason]
        assert zero_drops
        assert all(d.missing_fields == () for d in zero_drops)

    # ── shared fixtures for the openbb fake ──────────────────────────

    @staticmethod
    def _openbb_chains_df(spot: float = 100.0) -> pd.DataFrame:
        """Canonical synthetic OpenBB chain (two expiries, 3 strikes)."""
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

    @staticmethod
    def _fake_openbb_env(monkeypatch, chains_df, ticker_info) -> MagicMock:
        """Inject a fake ``openbb`` module and canned ``yfinance.Ticker``
        info (same pattern as tests/test_openbb.py)."""
        fake_openbb = types.ModuleType("openbb")
        fake_obb = MagicMock()
        fake_obb.derivatives.options.chains.return_value.to_df.return_value = chains_df
        fake_openbb.obb = fake_obb
        monkeypatch.setitem(sys.modules, "openbb", fake_openbb)

        class _FakeTicker:
            _INFO = ticker_info

            def __init__(self, symbol: str):
                self._symbol = symbol

            @property
            def info(self) -> dict:
                return dict(self._INFO.get(self._symbol, {}))

        monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
        return fake_obb


class TestFixB_YfinanceFallbackWarnings:
    """FIX B: yfinance fetch_chain logs explicit substitution warnings."""

    @staticmethod
    def _mock_fetch_chain(mock_ticker_class, irx_info, symbol_info) -> MagicMock:
        """Standard mock yfinance env; ^IRX info and symbol info canned."""

        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.info = symbol_info
        mock_ticker.options = ["2030-08-15", "2030-09-15"]

        mock_irx = MagicMock()
        mock_irx.info = irx_info
        mock_ticker_class.side_effect = lambda s: (
            mock_irx if s == "^IRX" else mock_ticker
        )

        strikes = [440, 450, 460]
        cols = {"strike": strikes, "lastPrice": [20, 15, 10],
                "bid": [19, 14, 9], "ask": [21, 16, 11],
                "volume": [100, 100, 100], "openInterest": [500, 500, 500]}
        mock_chain = MagicMock()
        mock_chain.calls = pd.DataFrame(cols | {"contractSymbol": ["c1", "c2", "c3"]})
        mock_chain.puts = pd.DataFrame(cols | {"contractSymbol": ["p1", "p2", "p3"]})
        mock_ticker.option_chain.return_value = mock_chain
        return mock_ticker

    @patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
    @patch("arbfree_vol.ingestion.yahoo.date")
    def test_r_warning_when_irx_fetch_fails(
        self, mock_date_class, mock_ticker_class, caplog
    ) -> None:
        """^IRX empty → WARNING names the symbol and substitutes r=0.05."""
        from datetime import date as real_date
        from arbfree_vol.ingestion.yahoo import fetch_chain

        self._mock_fetch_chain(
            mock_ticker_class,
            irx_info={},  # empty info → rate fetch fails
            symbol_info={"regularMarketPrice": 450.0},
        )
        today = real_date(2030, 7, 15)
        mock_date_class.today.return_value = today
        mock_date_class.fromisoformat.side_effect = real_date.fromisoformat

        with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.yahoo"):
            surface, _, _ = fetch_chain("SPY", max_expiries=2)

        assert surface.risk_free == 0.05
        assert "Risk-free rate unavailable for SPY" in caplog.text
        assert "substituting r=0.05" in caplog.text

    @patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
    @patch("arbfree_vol.ingestion.yahoo.date")
    def test_q_warning_when_dividend_yield_missing(
        self, mock_date_class, mock_ticker_class, caplog
    ) -> None:
        """dividendYield missing from ticker info → WARNING names the
        symbol and substitutes q=0.0."""
        from datetime import date as real_date
        from arbfree_vol.ingestion.yahoo import fetch_chain

        self._mock_fetch_chain(
            mock_ticker_class,
            irx_info={"regularMarketPrice": 4.85},  # r is fine
            symbol_info={"regularMarketPrice": 450.0},  # no dividendYield
        )
        today = real_date(2030, 7, 15)
        mock_date_class.today.return_value = today
        mock_date_class.fromisoformat.side_effect = real_date.fromisoformat

        with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.yahoo"):
            surface, _, _ = fetch_chain("SPY", max_expiries=2)

        assert surface.div_yield == 0.0
        assert "Dividend yield unavailable for SPY" in caplog.text
        assert "substituting q=0.0" in caplog.text


class TestFixB_OpenBBFallbackWarnings:
    """FIX B: openbb fetch_chain logs explicit substitution warnings."""

    def test_r_warning_when_irx_fetch_fails(self, monkeypatch, caplog) -> None:
        """^IRX empty → WARNING names the symbol and substitutes r=0.05."""
        TestFixA_IngestionPaths._fake_openbb_env(
            monkeypatch,
            TestFixA_IngestionPaths._openbb_chains_df(),
            {"^IRX": {}, "SPY": {"dividendYield": 0.011}},
        )

        with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.openbb"):
            surface, _, _ = openbb_mod.fetch_chain("SPY")

        assert surface.risk_free == 0.05
        assert "Risk-free rate unavailable for SPY" in caplog.text
        assert "substituting r=0.05" in caplog.text

    def test_q_warning_when_dividend_yield_missing(self, monkeypatch, caplog) -> None:
        """dividendYield missing from ticker info → WARNING names the
        symbol and substitutes q=0.0."""
        TestFixA_IngestionPaths._fake_openbb_env(
            monkeypatch,
            TestFixA_IngestionPaths._openbb_chains_df(),
            {"^IRX": {"regularMarketPrice": 5.0}, "SPY": {}},  # no dividendYield
        )

        with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.openbb"):
            surface, _, _ = openbb_mod.fetch_chain("SPY")

        assert surface.div_yield == 0.0
        assert "Dividend yield unavailable for SPY" in caplog.text
        assert "substituting q=0.0" in caplog.text
