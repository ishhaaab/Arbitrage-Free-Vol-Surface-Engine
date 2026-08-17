"""Tests for the yfinance fetcher (mocked, no network calls)."""

import logging
from unittest.mock import patch, MagicMock
from datetime import date

from pytest import approx

from arbfree_vol.models.surface import VolSurface
from arbfree_vol.models.option import OptionType


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
def test_fetch_chain_type(mock_ticker_class) -> None:
    """Smoke test: fetch_chain returns a VolSurface with structure."""
    import pandas as pd

    # Build a realistic mock
    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker

    # Tick info
    mock_ticker.info = {"regularMarketPrice": 450.0, "dividendYield": 0.005}

    # ^IRX info
    mock_irx = MagicMock()
    mock_irx.info = {"regularMarketPrice": 4.85}
    mock_ticker_class.side_effect = lambda s: (
        mock_irx if s == "^IRX" else mock_ticker
    )

    # Options
    today = date(2024, 7, 15)
    mock_ticker.options = [
        (today.replace(month=today.month + m)).isoformat()
        for m in range(1, 6)
    ]

    def _make_df(strikes, last_prices, bids, asks, otype: OptionType):
        rows = []
        for i, (K, lp, b, a) in enumerate(zip(strikes, last_prices, bids, asks)):
            rows.append({
                "strike": K,
                "lastPrice": lp,
                "bid": b,
                "ask": a,
                "volume": 100,
                "openInterest": 500,
                "contractSymbol": f"{otype.value}_{i}",
            })
        return pd.DataFrame(rows)

    strikes = [400, 420, 440, 450, 460, 480, 500]
    last = [55, 40, 22, 15, 9, 3, 1]
    bid = [54, 39, 21, 14, 8, 2, 0.5]
    ask = [56, 41, 23, 16, 10, 4, 1.5]

    calls_df = _make_df(strikes, last, bid, ask, OptionType.CALL)
    puts_df = _make_df(strikes, [53, 38, 20, 14, 10, 5, 3],
                       [52, 37, 19, 13, 9, 4, 2],
                       [54, 39, 21, 15, 11, 6, 4],
                       OptionType.PUT)

    mock_chain = MagicMock()
    mock_chain.calls = calls_df
    mock_chain.puts = puts_df
    mock_ticker.option_chain.return_value = mock_chain

    # Patch date.today to avoid expiry-time dependence
    with patch("arbfree_vol.ingestion.yahoo.date") as mock_date:
        mock_date.today.return_value = today
        mock_date.fromisoformat.side_effect = date.fromisoformat

        from arbfree_vol.ingestion.yahoo import fetch_chain
        surface, rejected, quality_drops = fetch_chain("SPY", max_expiries=2)

    assert isinstance(surface, VolSurface)
    assert isinstance(rejected, list)
    assert isinstance(quality_drops, list)
    assert surface.spot == 450.0
    assert len(surface.slices) == 2  # should get 2 weekly expiries


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
@patch("arbfree_vol.ingestion.yahoo.date")
def test_fetch_chain_falls_back_on_bad_rates(mock_date_class, mock_ticker_class, caplog) -> None:
    """When ^IRX or dividend yield is missing, the surface still builds
    with the documented fallbacks — and a WARNING states exactly which
    value was substituted and why (no silent defaults)."""
    import pandas as pd
    from datetime import date as real_date

    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    # No dividendYield but has a spot price
    mock_ticker.info = {"regularMarketPrice": 450.0}
    mock_ticker.options = ["2030-08-15", "2030-09-15"]

    # Patch date.today to a fixed date
    today = real_date(2030, 7, 15)
    mock_date_class.today.return_value = today
    mock_date_class.fromisoformat.side_effect = real_date.fromisoformat

    # ^IRX fails (empty info)
    mock_irx = MagicMock()
    mock_irx.info = {}
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

    from arbfree_vol.ingestion.yahoo import fetch_chain
    with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.yahoo"):
        surface, rejected, quality_drops = fetch_chain("SPY", max_expiries=2)

    assert surface.risk_free == 0.05  # default fallback — values unchanged
    assert surface.div_yield == 0.0  # default fallback — values unchanged
    # ...but the substitution must NOT be silent
    assert "substituting r=0.05" in caplog.text, (
        f"expected r fallback warning, got: {caplog.text}"
    )
    assert "substituting q=0.0" in caplog.text, (
        f"expected q fallback warning, got: {caplog.text}"
    )
    assert isinstance(rejected, list)
    assert isinstance(quality_drops, list)
    assert len(surface.slices) >= 1


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
@patch("arbfree_vol.ingestion.yahoo.date")
def test_fetch_chain_no_fallback_warning_when_rates_available(
    mock_date_class, mock_ticker_class, caplog,
) -> None:
    """When ^IRX and dividendYield ARE available, no fallback warning is
    logged — a regression guard against logging on every call."""
    import pandas as pd
    from datetime import date as real_date

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

    strikes = [440, 450, 460]
    cols = {"strike": strikes, "lastPrice": [20, 15, 10],
            "bid": [19, 14, 9], "ask": [21, 16, 11],
            "volume": [100, 100, 100], "openInterest": [500, 500, 500]}
    mock_chain = MagicMock()
    mock_chain.calls = pd.DataFrame(cols | {"contractSymbol": ["c1", "c2", "c3"]})
    mock_chain.puts = pd.DataFrame(cols | {"contractSymbol": ["p1", "p2", "p3"]})
    mock_ticker.option_chain.return_value = mock_chain

    from arbfree_vol.ingestion.yahoo import fetch_chain
    with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.yahoo"):
        surface, rejected, quality_drops = fetch_chain("SPY", max_expiries=2)

    assert surface.risk_free == approx(0.0485)
    assert surface.div_yield == approx(0.005)
    assert "substituting" not in caplog.text, (
        f"no fallback should be logged with real rates, got: {caplog.text}"
    )


def _mock_index_chain(mock_ticker_class, symbol="^SPX"):
    """Shared mock chain for index-symbol fetch_chain tests."""
    import pandas as pd

    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.info = {"regularMarketPrice": 450.0}
    mock_ticker.options = ["2030-08-15", "2030-09-15"]

    mock_irx = MagicMock()
    mock_irx.info = {"regularMarketPrice": 4.85}
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
def test_fetch_chain_observed_zero_dividend_yield_is_preserved(
    mock_date_class, mock_ticker_class, caplog,
) -> None:
    """An OBSERVED zero dividend yield (dividendYield present as 0.0 in
    the ticker info) must be preserved as q=0.0 — and the warning must
    say the yield was observed as zero, NOT that it was missing and
    substituted (the pre-fix bug conflated q==0 with missing)."""
    import pandas as pd
    from datetime import date as real_date

    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    # dividendYield present with value 0.0: observed zero, not missing
    mock_ticker.info = {"regularMarketPrice": 450.0, "dividendYield": 0.0}
    mock_ticker.options = ["2030-08-15", "2030-09-15"]

    today = real_date(2030, 7, 15)
    mock_date_class.today.return_value = today
    mock_date_class.fromisoformat.side_effect = real_date.fromisoformat

    mock_irx = MagicMock()
    mock_irx.info = {"regularMarketPrice": 4.85}
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

    from arbfree_vol.ingestion.yahoo import fetch_chain
    with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.yahoo"):
        surface, _, _ = fetch_chain("SPY", max_expiries=2)

    # q=0.0 is used as OBSERVED (the value is unchanged), and the warning
    # reflects the observation rather than a substitution.
    assert surface.div_yield == 0.0
    assert "observed as zero" in caplog.text, (
        f"expected the observed-zero warning, got: {caplog.text}"
    )
    assert "substituting q=0.0" not in caplog.text, (
        f"an observed zero must NOT be logged as a substitution, got: "
        f"{caplog.text}"
    )


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
@patch("arbfree_vol.ingestion.yahoo.date")
def test_fetch_chain_index_q_mix_is_logged(
    mock_date_class, mock_ticker_class, monkeypatch, caplog,
) -> None:
    """When some index slices get a per-expiry parity q and others do
    not, the resulting MIXED q-quality surface is logged explicitly."""
    from datetime import date as real_date
    import arbfree_vol.ingestion.yahoo as yf_mod

    _mock_index_chain(mock_ticker_class)
    today = real_date(2030, 7, 15)
    mock_date_class.today.return_value = today
    mock_date_class.fromisoformat.side_effect = real_date.fromisoformat

    # First slice: parity succeeds.  Second slice: parity fails.
    calls = {"n": 0}

    def fake_estimate(sl, spot, r):
        calls["n"] += 1
        return 0.01 if calls["n"] == 1 else None

    monkeypatch.setattr(
        "arbfree_vol.ingestion._index_rates._estimate_index_dividend_yield",
        fake_estimate,
    )

    with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.yahoo"):
        surface, _, _ = yf_mod.fetch_chain("^SPX", max_expiries=2)

    assert "MIXED" in caplog.text, (
        f"expected MIXED q-quality warning, got: {caplog.text}"
    )
    assert "1/2" in caplog.text
    # Values unchanged: first slice has its own parity q, second falls
    # back to the surface q (the parity median here)
    assert surface.slices[0].div_yield == approx(0.01)
    assert surface.slices[1].div_yield is None
    assert surface.div_yield == approx(0.01)


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
@patch("arbfree_vol.ingestion.yahoo.date")
def test_fetch_chain_index_q_all_fail_etf_fallback_logged(
    mock_date_class, mock_ticker_class, monkeypatch, caplog,
) -> None:
    """All slices fail parity -> representative ETF yield fallback is
    logged, and the surface carries the ETF q."""
    from datetime import date as real_date
    import arbfree_vol.ingestion.yahoo as yf_mod

    _mock_index_chain(mock_ticker_class)
    today = real_date(2030, 7, 15)
    mock_date_class.today.return_value = today
    mock_date_class.fromisoformat.side_effect = real_date.fromisoformat

    monkeypatch.setattr(
        "arbfree_vol.ingestion._index_rates._estimate_index_dividend_yield",
        lambda sl, spot, r: None,
    )
    monkeypatch.setattr(
        "arbfree_vol.ingestion._index_rates._get_representative_dividend_yield",
        lambda symbol: 0.013,
    )

    with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.yahoo"):
        surface, _, _ = yf_mod.fetch_chain("^SPX", max_expiries=2)

    assert "representative ETF yield" in caplog.text, (
        f"expected ETF fallback warning, got: {caplog.text}"
    )
    assert surface.div_yield == approx(0.013)
    for sl in surface.slices:
        assert sl.div_yield is None  # per-slice q untouched


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
@patch("arbfree_vol.ingestion.yahoo.date")
def test_fetch_chain_index_representative_zero_yield_preserved(
    mock_date_class, mock_ticker_class, monkeypatch, caplog,
) -> None:
    """All slices fail parity and the representative ETF's dividend yield
    is PRESENT as 0.0: the surface q must be 0.0 AS OBSERVED (not
    treated as missing), the representative helper must log "observed as
    zero", and fetch_chain must NOT claim the representative yield was
    unavailable or hardcoded.

    Regression (pre-fix): ``_get_representative_dividend_yield``
    required ``q > 0``, so a present-zero representative yield returned
    None and fetch_chain logged "no representative ETF yield available;
    surface q hardcoded to 0.0" — conflating an observed zero with
    missing data (commit 5bf429a aligned the primary paths; this pins
    the same semantics on the representative fallback)."""
    import pandas as pd
    from datetime import date as real_date
    import arbfree_vol.ingestion.yahoo as yf_mod

    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    # The same mock serves as the ^SPX chain ticker AND the
    # representative SPY ticker (the helper calls yf.Ticker("SPY")); its
    # info carries a PRESENT zero dividendYield.
    mock_ticker.info = {"regularMarketPrice": 450.0, "dividendYield": 0.0}
    mock_ticker.options = ["2030-08-15", "2030-09-15"]

    today = real_date(2030, 7, 15)
    mock_date_class.today.return_value = today
    mock_date_class.fromisoformat.side_effect = real_date.fromisoformat

    mock_irx = MagicMock()
    mock_irx.info = {"regularMarketPrice": 4.85}
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

    # Parity estimation fails for all slices so the representative
    # fallback path runs with the REAL helper (not monkeypatched).
    monkeypatch.setattr(
        "arbfree_vol.ingestion._index_rates._estimate_index_dividend_yield",
        lambda sl, spot, r: None,
    )

    with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.yahoo"):
        surface, _, _ = yf_mod.fetch_chain("^SPX", max_expiries=2)

    assert surface.div_yield == 0.0
    # The helper's observed-zero provenance is logged with its exact
    # wording (unique to the representative path).
    assert "representative ETF SPY has dividendYield present as 0.0" in caplog.text, (
        f"expected the representative observed-zero warning, got: {caplog.text}"
    )
    assert "no representative ETF yield available" not in caplog.text, (
        f"a present-zero representative yield must not be logged as "
        f"unavailable, got: {caplog.text}"
    )
    assert "hardcoded to 0.0" not in caplog.text, (
        f"a present-zero representative yield must not be logged as a "
        f"hardcoded substitution, got: {caplog.text}"
    )


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
@patch("arbfree_vol.ingestion.yahoo.date")
def test_fetch_chain_index_q_all_fail_zero_fallback_logged(
    mock_date_class, mock_ticker_class, monkeypatch, caplog,
) -> None:
    """All slices fail parity AND no representative ETF -> q=0.0
    fallback-of-last-resort is logged explicitly."""
    from datetime import date as real_date
    import arbfree_vol.ingestion.yahoo as yf_mod

    _mock_index_chain(mock_ticker_class)
    today = real_date(2030, 7, 15)
    mock_date_class.today.return_value = today
    mock_date_class.fromisoformat.side_effect = real_date.fromisoformat

    monkeypatch.setattr(
        "arbfree_vol.ingestion._index_rates._estimate_index_dividend_yield",
        lambda sl, spot, r: None,
    )
    monkeypatch.setattr(
        "arbfree_vol.ingestion._index_rates._get_representative_dividend_yield",
        lambda symbol: None,
    )

    with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion.yahoo"):
        surface, _, _ = yf_mod.fetch_chain("^SPX", max_expiries=2)

    assert "hardcoded to 0.0" in caplog.text, (
        f"expected q=0.0 last-resort warning, got: {caplog.text}"
    )
    assert surface.div_yield == 0.0


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
@patch("arbfree_vol.ingestion.yahoo.date")
def test_fetch_chain_disable_quality_filter(mock_date_class, mock_ticker_class) -> None:
    """disable_quality_filter=True skips the filter and returns empty drops."""
    import pandas as pd
    from datetime import date as real_date

    mock_ticker = MagicMock()
    mock_ticker_class.return_value = mock_ticker
    mock_ticker.info = {"regularMarketPrice": 450.0, "dividendYield": 0.005}
    mock_ticker.options = ["2030-08-15"]

    today = real_date(2030, 7, 15)
    mock_date_class.today.return_value = today
    mock_date_class.fromisoformat.side_effect = real_date.fromisoformat

    mock_irx = MagicMock()
    mock_irx.info = {"regularMarketPrice": 4.85}
    mock_ticker_class.side_effect = lambda s: (
        mock_irx if s == "^IRX" else mock_ticker
    )

    # Include strikes with low OI that would normally be filtered
    strikes = [440, 450, 460]
    cols = {
        "strike": strikes,
        "lastPrice": [20, 15, 10],
        "bid": [19, 14, 9],
        "ask": [21, 16, 11],
        "volume": [100, 100, 100],
        "openInterest": [1, 500, 1],  # mixed: two low OI, one passes filter
    }
    mock_chain = MagicMock()
    mock_chain.calls = pd.DataFrame(cols | {"contractSymbol": ["c1", "c2", "c3"]})
    mock_chain.puts = pd.DataFrame(cols | {"contractSymbol": ["p1", "p2", "p3"]})
    mock_ticker.option_chain.return_value = mock_chain

    from arbfree_vol.ingestion.yahoo import fetch_chain

    # With filter disabled: low-OI strikes pass through, quality_drops is empty
    surface_raw, _, quality_drops_raw = fetch_chain(
        "SPY", max_expiries=1, disable_quality_filter=True
    )
    assert isinstance(quality_drops_raw, list)
    assert len(quality_drops_raw) == 0  # no drops when filter is disabled

    # With filter enabled (default): low-OI strikes get dropped
    surface_filtered, _, quality_drops_filtered = fetch_chain(
        "SPY", max_expiries=1, disable_quality_filter=False
    )
    assert len(quality_drops_filtered) > 0  # OI=1 < 10 → dropped
