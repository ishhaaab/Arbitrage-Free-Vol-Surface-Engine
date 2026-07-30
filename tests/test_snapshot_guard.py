"""Tests for the snapshot-time guard."""

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from arbfree_vol.data.snapshot_guard import check_snapshot_time


_EASTERN = ZoneInfo("US/Eastern")


class TestSnapshotGuard:
    """Tests for check_snapshot_time."""

    def test_snapshot_guard_passes_during_safe_window(self):
        """A snapshot at 12:00 ET on a normal weekday returns None."""
        # 2026-07-15 is a Wednesday, not in the exclusion set
        dt = datetime(2026, 7, 15, 12, 0, 0, tzinfo=_EASTERN)
        result = check_snapshot_time(now=dt)
        assert result is None

    def test_snapshot_guard_warns_outside_window(self):
        """A snapshot before the safe window returns a warning string."""
        # 08:00 ET — before 10:30
        dt = datetime(2026, 7, 15, 8, 0, 0, tzinfo=_EASTERN)
        result = check_snapshot_time(now=dt)
        assert result is not None
        assert "outside safe window" in result

    def test_snapshot_guard_warns_after_close(self):
        """A snapshot after the safe window returns a warning string."""
        # 16:00 ET — after 15:30
        dt = datetime(2026, 7, 15, 16, 0, 0, tzinfo=_EASTERN)
        result = check_snapshot_time(now=dt)
        assert result is not None
        assert "outside safe window" in result

    def test_snapshot_guard_warns_on_exclusion_date(self):
        """A snapshot on a known event day returns a warning string."""
        from arbfree_vol.data.snapshot_guard import _EXCLUSION_DATES
        # Pick a date that IS in the exclusion set
        sample_date_str = "2026-07-29"  # FOMC day
        if sample_date_str in _EXCLUSION_DATES:
            dt = datetime(2026, 7, 29, 12, 0, 0, tzinfo=_EASTERN)
            result = check_snapshot_time(now=dt)
            assert result is not None
            assert "event day" in result
        else:
            pytest.skip("2026-07-29 not in exclusion set")

    def test_snapshot_guard_warns_on_known_exclusion_date(self):
        """Use a date that is definitely in the exclusion set."""
        from arbfree_vol.data.snapshot_guard import _EXCLUSION_DATES
        # Pick the first exclusion date
        sample_date_str = next(iter(_EXCLUSION_DATES))
        y, m, d = map(int, sample_date_str.split("-"))
        # Use a time inside the safe window so only the date check triggers
        dt = datetime(y, m, d, 12, 0, 0, tzinfo=_EASTERN)
        result = check_snapshot_time(now=dt)
        assert result is not None
        assert "event day" in result
        assert sample_date_str in result

    def test_snapshot_guard_naive_time_gets_localised(self):
        """A naive datetime is treated as US/Eastern."""
        dt = datetime(2026, 7, 15, 12, 0, 0)  # no tzinfo
        result = check_snapshot_time(now=dt)
        # Should be treated as ET and pass (not an event day)
        assert result is None

    def test_snapshot_guard_custom_exclusion_dates(self):
        """Custom exclusion dates override the built-in set."""
        dt = datetime(2026, 3, 15, 12, 0, 0, tzinfo=_EASTERN)
        # Without custom dates — should pass (not an event day normally)
        result_normal = check_snapshot_time(now=dt, exclusion_dates=set())
        assert result_normal is None

        # With custom exclusion
        result_custom = check_snapshot_time(
            now=dt, exclusion_dates={"2026-03-15"}
        )
        assert result_custom is not None
        assert "2026-03-15" in result_custom

    def test_snapshot_guard_boundary_at_start(self):
        """Exactly at safe_window_start (10:30) should pass."""
        dt = datetime(2026, 7, 15, 10, 30, 0, tzinfo=_EASTERN)
        result = check_snapshot_time(now=dt)
        assert result is None

    def test_snapshot_guard_boundary_at_end(self):
        """Exactly at safe_window_end (15:30) should pass."""
        dt = datetime(2026, 7, 15, 15, 30, 0, tzinfo=_EASTERN)
        result = check_snapshot_time(now=dt)
        assert result is None

    def test_snapshot_guard_one_minute_before_start(self):
        """One minute before safe_window_start should warn."""
        dt = datetime(2026, 7, 15, 10, 29, 0, tzinfo=_EASTERN)
        result = check_snapshot_time(now=dt)
        assert result is not None
        assert "outside safe window" in result
