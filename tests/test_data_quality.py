"""Tests for the data quality filter."""

import pandas as pd
import pytest

from arbfree_vol.data.quality import (
    DataQualityConfig,
    DropRecord,
    filter_option_chain,
)


def _make_chain_df(strikes, oi, volume, bid, ask):
    """Build a minimal option chain DataFrame."""
    return pd.DataFrame({
        "strike": strikes,
        "openInterest": oi,
        "volume": volume,
        "bid": bid,
        "ask": ask,
    })


class TestFilterOptionChain:
    """Tests for filter_option_chain."""

    def test_all_pass_default_config(self):
        """Strikes above all thresholds pass through unchanged."""
        df = _make_chain_df(
            strikes=[100, 110, 120],
            oi=[100, 200, 300],
            volume=[10, 20, 30],
            bid=[9.0, 4.5, 1.8],
            ask=[11.0, 5.5, 2.2],  # all spreads under 50%
        )
        filtered, drops = filter_option_chain(df, "2026-08-15")
        assert len(filtered) == 3
        assert len(drops) == 0

    def test_low_oi_dropped(self):
        """Strikes with OI below min_open_interest are dropped."""
        df = _make_chain_df(
            strikes=[100, 110, 120],
            oi=[5, 200, 3],  # 100 and 120 have low OI
            volume=[10, 20, 30],
            bid=[9.0, 4.0, 1.0],
            ask=[11.0, 6.0, 2.0],
        )
        config = DataQualityConfig(min_open_interest=10)
        filtered, drops = filter_option_chain(df, "2026-08-15", config)
        assert len(filtered) == 1
        assert len(drops) == 2
        assert filtered.iloc[0]["strike"] == 110

    def test_low_volume_dropped(self):
        """Strikes with volume below min_volume are dropped."""
        df = _make_chain_df(
            strikes=[100, 110, 120],
            oi=[100, 200, 300],
            volume=[0, 5, 0],  # 100 and 120 have zero volume
            bid=[9.0, 4.0, 1.0],
            ask=[11.0, 6.0, 2.0],
        )
        config = DataQualityConfig(min_volume=1)
        filtered, drops = filter_option_chain(df, "2026-08-15", config)
        assert len(filtered) == 1
        assert len(drops) == 2
        assert filtered.iloc[0]["strike"] == 110

    def test_wide_spread_dropped(self):
        """Strikes with bid-ask spread above max_bid_ask_pct are dropped."""
        df = _make_chain_df(
            strikes=[100, 110, 120],
            oi=[100, 200, 300],
            volume=[10, 20, 30],
            bid=[5.0, 4.5, 1.8],
            ask=[15.0, 5.5, 2.2],  # 100: spread=10/10=100% > 50%, others pass
        )
        config = DataQualityConfig(max_bid_ask_pct=50.0)
        filtered, drops = filter_option_chain(df, "2026-08-15", config)
        assert len(filtered) == 2
        assert len(drops) == 1
        assert drops[0].strike == 100

    def test_drop_reason_contains_oi(self):
        """Drop reason string includes the OI violation."""
        df = _make_chain_df(
            strikes=[100],
            oi=[3],
            volume=[10],
            bid=[9.0],
            ask=[11.0],
        )
        config = DataQualityConfig(min_open_interest=10)
        _, drops = filter_option_chain(df, "2026-08-15", config)
        assert len(drops) == 1
        assert "OI=3<10" in drops[0].reason

    def test_drop_reason_contains_volume(self):
        """Drop reason string includes the volume violation."""
        df = _make_chain_df(
            strikes=[100],
            oi=[100],
            volume=[0],
            bid=[9.0],
            ask=[11.0],
        )
        config = DataQualityConfig(min_volume=1)
        _, drops = filter_option_chain(df, "2026-08-15", config)
        assert len(drops) == 1
        assert "vol=0<1" in drops[0].reason

    def test_drop_reason_contains_spread(self):
        """Drop reason string includes the spread violation."""
        df = _make_chain_df(
            strikes=[100],
            oi=[100],
            volume=[10],
            bid=[5.0],
            ask=[15.0],
        )
        config = DataQualityConfig(max_bid_ask_pct=50.0)
        _, drops = filter_option_chain(df, "2026-08-15", config)
        assert len(drops) == 1
        assert "spread=" in drops[0].reason
        assert ">50.0%" in drops[0].reason

    def test_multiple_violations_compound_reason(self):
        """A strike failing multiple thresholds has all reasons listed."""
        df = _make_chain_df(
            strikes=[100],
            oi=[3],
            volume=[0],
            bid=[5.0],
            ask=[15.0],
        )
        config = DataQualityConfig(
            min_open_interest=10, min_volume=1, max_bid_ask_pct=50.0
        )
        _, drops = filter_option_chain(df, "2026-08-15", config)
        assert len(drops) == 1
        assert "OI=3<10" in drops[0].reason
        assert "vol=0<1" in drops[0].reason
        assert "spread=" in drops[0].reason

    def test_expiry_recorded_in_drop(self):
        """The expiry string is passed through to DropRecord."""
        df = _make_chain_df(
            strikes=[100],
            oi=[3],
            volume=[10],
            bid=[9.0],
            ask=[11.0],
        )
        _, drops = filter_option_chain(df, "2030-09-15")
        assert drops[0].expiry == "2030-09-15"

    def test_empty_df_returns_empty(self):
        """An empty DataFrame returns empty filtered and empty drops."""
        df = _make_chain_df(strikes=[], oi=[], volume=[], bid=[], ask=[])
        filtered, drops = filter_option_chain(df, "2026-08-15")
        assert len(filtered) == 0
        assert len(drops) == 0
