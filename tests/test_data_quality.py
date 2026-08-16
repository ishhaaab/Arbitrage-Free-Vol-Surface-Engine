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

    def test_low_volume_no_longer_filtered(self):
        """Volume is NOT a filter criterion — zero-volume strikes pass."""
        df = _make_chain_df(
            strikes=[100, 110, 120],
            oi=[100, 200, 300],
            volume=[0, 5, 0],  # 100 and 120 have zero volume
            bid=[9.0, 4.5, 1.8],
            ask=[11.0, 5.5, 2.2],  # all spreads well under 50%
        )
        # No min_volume field exists — all strikes pass the volume check.
        config = DataQualityConfig(min_open_interest=10)
        filtered, drops = filter_option_chain(df, "2026-08-15", config)
        assert len(filtered) == 3  # all three pass
        assert len(drops) == 0

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

    def test_volume_still_recorded_in_drop(self):
        """Volume is recorded in DropRecord for diagnostic context, not as a reason."""
        df = _make_chain_df(
            strikes=[100],
            oi=[100],
            volume=[0],
            bid=[5.0],
            ask=[15.0],  # wide spread — this is the actual filter hit
        )
        config = DataQualityConfig(max_bid_ask_pct=50.0)
        _, drops = filter_option_chain(df, "2026-08-15", config)
        assert len(drops) == 1
        # Volume is NOT in the reason — only spread is
        assert "vol" not in drops[0].reason
        assert "spread=" in drops[0].reason
        # But volume IS recorded for diagnostic purposes
        assert drops[0].volume == 0

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
            min_open_interest=10, max_bid_ask_pct=50.0
        )
        _, drops = filter_option_chain(df, "2026-08-15", config)
        assert len(drops) == 1
        assert "OI=3<10" in drops[0].reason
        assert "spread=" in drops[0].reason
        # Volume is recorded but NOT a reason
        assert drops[0].volume == 0
        assert "vol" not in drops[0].reason

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

    def test_zero_volume_high_oi_tight_spread_passes(self):
        """A zero-volume strike with high OI and tight spread is NOT dropped.

        This is the key invariant: volume=0 is normal for legitimate
        market-maker quotes away from ATM and must not be filtered.
        """
        df = _make_chain_df(
            strikes=[100],
            oi=[100],       # well above min_open_interest=10
            volume=[0],     # zero volume — must still pass
            bid=[0.99],     # tight spread: (1.01-0.99)/1.00 = 2%
            ask=[1.01],
        )
        filtered, drops = filter_option_chain(df, "2026-08-15")
        assert len(filtered) == 1
        assert len(drops) == 0
        assert filtered.iloc[0]["strike"] == 100

    def test_missing_oi_distinguishable_from_zero(self):
        """Missing/NaN OI is dropped with a 'missing' reason and flag —
        never byte-identical to a genuinely observed OI=0."""
        df = _make_chain_df(
            strikes=[100, 110],
            oi=[None, 0],  # 100: OI missing; 110: OI genuinely zero
            volume=[10, 10],
            bid=[9.0, 9.0],
            ask=[11.0, 11.0],
        )
        _, drops = filter_option_chain(df, "2026-08-15")
        assert len(drops) == 2
        by_strike = {d.strike: d for d in drops}
        missing = by_strike[100.0]
        zero = by_strike[110.0]
        assert "OI=missing<10" in missing.reason
        assert missing.missing_fields == ("open_interest",)
        assert "OI=0<10" in zero.reason
        assert zero.missing_fields == ()

    def test_missing_oi_via_nan_flagged(self):
        """NaN openInterest is treated as missing, not zero."""
        df = _make_chain_df(
            strikes=[100],
            oi=[float("nan")],
            volume=[10],
            bid=[9.0],
            ask=[11.0],
        )
        _, drops = filter_option_chain(df, "2026-08-15")
        assert len(drops) == 1
        assert "OI=missing<10" in drops[0].reason
        assert drops[0].missing_fields == ("open_interest",)

    def test_missing_bid_or_ask_distinguishable(self):
        """A missing bid (or ask) with the other side present produces a
        spread drop that names the missing side — not a fabricated
        spread number indistinguishable from a real quote pair, and not
        a silent pass-through with a made-up mid."""
        df = _make_chain_df(
            strikes=[100, 110, 120],
            oi=[100, 100, 100],
            volume=[10, 10, 10],
            bid=[float("nan"), 5.0, float("nan")],
            ask=[10.0, float("nan"), float("nan")],
        )
        _, drops = filter_option_chain(df, "2026-08-15")
        by_strike = {d.strike: d for d in drops}
        # bid missing, ask present → dropped, names the missing side
        assert "missing: bid" in by_strike[100.0].reason
        assert by_strike[100.0].missing_fields == ("bid",)
        # ask missing, bid present → previously passed with a fabricated
        # mid=bid/2 and a negative spread; now flagged instead
        assert "missing: ask" in by_strike[110.0].reason
        assert by_strike[110.0].missing_fields == ("ask",)
        # both sides missing → dropped, naming both sides (mid = 0 must
        # not skip the spread branch — a no-quote strike is the filter's
        # documented target)
        assert "missing: bid, ask" in by_strike[120.0].reason
        assert by_strike[120.0].missing_fields == ("bid", "ask")

    def test_missing_volume_recorded_not_reason(self):
        """Missing volume is recorded in missing_fields but is not a
        filter reason (volume is never a criterion)."""
        df = _make_chain_df(
            strikes=[100],
            oi=[3],  # OI violation drives the drop
            volume=[None],
            bid=[9.0],
            ask=[11.0],
        )
        _, drops = filter_option_chain(df, "2026-08-15")
        assert len(drops) == 1
        assert "volume" in drops[0].missing_fields
        assert "vol" not in drops[0].reason

    def test_absent_oi_column_treated_as_missing(self):
        """A DataFrame with NO ``openInterest`` column (the provider
        omitted the field entirely) must flag every row's OI as MISSING
        — ``OI=missing<10`` with ``missing_fields=("open_interest",)`` —
        never as an observed zero.

        The pre-fix ``row.get("openInterest", 0)`` default conflated an
        absent column with a real 0: a mass drop caused by a provider
        omitting a column was mislabelled ``OI=0<10`` (illiquid strike)
        instead of the honest missing-field reason."""
        df = pd.DataFrame({
            "strike": [100.0],
            "volume": [10],
            "bid": [9.0],
            "ask": [11.0],
            # no "openInterest" column at all
        })
        _, drops = filter_option_chain(df, "2026-08-15")
        assert len(drops) == 1
        assert "OI=missing<10" in drops[0].reason
        assert drops[0].missing_fields == ("open_interest",)

    def test_absent_bid_column_treated_as_missing(self):
        """A DataFrame with NO ``bid`` column flags the bid side as
        missing (one-sided quote drop naming 'bid') — not a fabricated
        200% spread computed from an absent side treated as an observed
        zero bid."""
        df = pd.DataFrame({
            "strike": [100.0],
            "openInterest": [100],
            "volume": [10],
            "ask": [10.0],
            # no "bid" column at all
        })
        _, drops = filter_option_chain(df, "2026-08-15")
        assert len(drops) == 1
        assert "spread=missing (missing: bid)" in drops[0].reason
        assert drops[0].missing_fields == ("bid",)

    def test_absent_column_distinct_from_present_zero(self):
        """Side by side, an ABSENT OI column and a PRESENT OI=0 produce
        distinct DropRecords — absent is missing, present-zero is an
        observed zero."""
        # Two DataFrames cannot share columns with different presence in
        # one call, so pin the two outcomes separately and assert they
        # differ in exactly the documented way.
        df_zero = _make_chain_df(
            strikes=[100.0], oi=[0], volume=[10], bid=[9.0], ask=[11.0],
        )
        _, drops_zero = filter_option_chain(df_zero, "2026-08-15")
        assert "OI=0<10" in drops_zero[0].reason
        assert drops_zero[0].missing_fields == ()

        df_absent = pd.DataFrame({
            "strike": [100.0], "volume": [10], "bid": [9.0], "ask": [11.0],
        })
        _, drops_absent = filter_option_chain(df_absent, "2026-08-15")
        assert "OI=missing<10" in drops_absent[0].reason
        assert drops_absent[0].missing_fields == ("open_interest",)
