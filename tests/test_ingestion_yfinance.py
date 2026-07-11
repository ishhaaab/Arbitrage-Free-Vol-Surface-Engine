"""Tests for the yfinance fetcher (mocked, no network calls)."""

from unittest.mock import patch, MagicMock
from datetime import date

from pytest import approx

from arbfree_vol.models.surface import VolSurface
from arbfree_vol.models.option import OptionType
from arbfree_vol.ingestion.cleaning import RejectionRecord


@patch("arbfree_vol.ingestion.yfinance.yf.Ticker")
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
    with patch("arbfree_vol.ingestion.yfinance.date") as mock_date:
        mock_date.today.return_value = today
        mock_date.fromisoformat.side_effect = date.fromisoformat

        from arbfree_vol.ingestion.yfinance import fetch_chain
        surface, rejected = fetch_chain("SPY", max_expiries=2)

    assert isinstance(surface, VolSurface)
    assert isinstance(rejected, list)
    assert surface.spot == 450.0
    assert len(surface.slices) == 2  # should get 2 weekly expiries


@patch("arbfree_vol.ingestion.yfinance.yf.Ticker")
@patch("arbfree_vol.ingestion.yfinance.date")
def test_fetch_chain_falls_back_on_bad_rates(mock_date_class, mock_ticker_class) -> None:
    """When ^IRX or dividend yield is missing, the surface still builds.

    The forward pre-pass (detect_with_forward) handles the correction.
    """
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
            "bid": [19, 14, 9], "ask": [21, 16, 11]}
    mock_chain = MagicMock()
    mock_chain.calls = pd.DataFrame(cols | {"contractSymbol": ["c1", "c2", "c3"]})
    mock_chain.puts = pd.DataFrame(cols | {"contractSymbol": ["p1", "p2", "p3"]})
    mock_ticker.option_chain.return_value = mock_chain

    from arbfree_vol.ingestion.yfinance import fetch_chain
    surface, rejected = fetch_chain("SPY", max_expiries=2)

    assert surface.risk_free == 0.05  # default fallback
    assert surface.div_yield == 0.0  # default fallback
    assert isinstance(rejected, list)
    assert len(surface.slices) >= 1
