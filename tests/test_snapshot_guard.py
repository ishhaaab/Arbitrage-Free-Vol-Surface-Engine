"""Tests for the snapshot-time guard."""

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

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
        """A snapshot on a known event day returns a warning string.

        The date is hard-coded from the built-in ``_EXCLUSION_DATES``
        set, and its membership is asserted unconditionally: if the set
        ever lost this date (or became empty), the test FAILS loudly
        instead of skipping.
        """
        from arbfree_vol.data.snapshot_guard import _EXCLUSION_DATES
        sample_date_str = "2026-07-29"  # FOMC day
        assert sample_date_str in _EXCLUSION_DATES, (
            f"{sample_date_str} must be in _EXCLUSION_DATES for this "
            "test to be meaningful"
        )
        dt = datetime(2026, 7, 29, 12, 0, 0, tzinfo=_EASTERN)
        result = check_snapshot_time(now=dt)
        assert result is not None
        assert "event day" in result
        assert sample_date_str in result

    def test_snapshot_guard_warns_on_known_exclusion_date(self):
        """A second hard-coded exclusion date (CPI day) also warns, with
        the date named in the message."""
        from arbfree_vol.data.snapshot_guard import _EXCLUSION_DATES
        sample_date_str = "2026-08-12"  # CPI day
        assert sample_date_str in _EXCLUSION_DATES, (
            f"{sample_date_str} must be in _EXCLUSION_DATES for this "
            "test to be meaningful"
        )
        # Use a time inside the safe window so only the date check triggers
        y, m, d = map(int, sample_date_str.split("-"))
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
        """Custom exclusion dates override the built-in set.

        The base date is a WEEKDAY (2026-03-16 is a Monday) so the only
        thing that can trigger a warning under custom dates is the
        custom exclusion set — the weekend check must not interfere with
        this test's intent (2026-03-15, the old fixture, is a Sunday and
        is now caught by the weekend check)."""
        dt = datetime(2026, 3, 16, 12, 0, 0, tzinfo=_EASTERN)
        assert dt.weekday() == 0, "test setup error: 2026-03-16 must be Monday"
        # Without custom dates — should pass (not an event day normally)
        result_normal = check_snapshot_time(now=dt, exclusion_dates=set())
        assert result_normal is None

        # With custom exclusion
        result_custom = check_snapshot_time(
            now=dt, exclusion_dates={"2026-03-16"}
        )
        assert result_custom is not None
        assert "2026-03-16" in result_custom

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

    def test_snapshot_guard_warns_on_weekend_inside_window(self):
        """A Saturday at 12:00 ET (INSIDE the clock window) must still
        warn: the market is closed all day on weekends, so the weekend
        check fires regardless of clock time.  2026-07-18 is a Saturday
        (2026-07-15 is the Wednesday used by the weekday tests)."""
        # 2026-07-18 is a Saturday (2026-07-15 is Wednesday).
        sat = datetime(2026, 7, 18, 12, 0, 0, tzinfo=_EASTERN)
        assert sat.weekday() == 5, "test setup error: 2026-07-18 must be Saturday"
        result = check_snapshot_time(now=sat)
        assert result is not None
        assert "weekend" in result

    def test_snapshot_guard_warns_on_weekend_outside_window(self):
        """A Sunday at 08:00 ET (OUTSIDE the clock window) also warns:
        the weekend check fires before the clock-window check, so the
        documented contract (warns on weekends regardless of clock time)
        holds in both the inside-window and outside-window cases."""
        # 2026-07-19 is a Sunday.
        sun = datetime(2026, 7, 19, 8, 0, 0, tzinfo=_EASTERN)
        assert sun.weekday() == 6, "test setup error: 2026-07-19 must be Sunday"
        result = check_snapshot_time(now=sun)
        assert result is not None
        assert "weekend" in result

    def test_snapshot_guard_weekday_inside_window_no_warning(self):
        """A weekday inside the clock window (Wednesday 12:00 ET, not an
        event date) produces NO warning — the weekend check must not
        false-positive on weekdays."""
        wed = datetime(2026, 7, 15, 12, 0, 0, tzinfo=_EASTERN)
        assert wed.weekday() == 2, "test setup error: 2026-07-15 must be Wednesday"
        result = check_snapshot_time(now=wed)
        assert result is None
