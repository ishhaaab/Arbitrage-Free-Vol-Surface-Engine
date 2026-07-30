"""Snapshot-time guard for option chain fetching.

Warns when fetching data outside a "safe window" (US/Eastern market
hours) or on known event days (FOMC, CPI, opex).  The guard does NOT
block — it returns a warning string and the caller decides what to do.

Usage::

    from arbfree_vol.data.snapshot_guard import check_snapshot_time

    warning = check_snapshot_time()
    if warning:
        warnings.warn(warning)
"""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

# ── Exclusion dates ──────────────────────────────────────────────────
# Known event days where option data may be unreliable.  A small
# hard-coded list is fine — no external calendar API needed.
# Format: YYYY-MM-DD strings.

_EXCLUSION_DATES: set[str] = {
    # 2025
    "2025-01-29",  # FOMC
    "2025-03-19",  # FOMC
    "2025-05-07",  # FOMC
    "2025-06-18",  # FOMC
    "2025-07-30",  # FOMC
    "2025-09-17",  # FOMC
    "2025-10-29",  # FOMC
    "2025-12-17",  # FOMC
    "2025-01-14",  # CPI
    "2025-02-12",  # CPI
    "2025-03-12",  # CPI
    "2025-04-10",  # CPI
    "2025-05-13",  # CPI
    "2025-06-11",  # CPI
    "2025-07-10",  # CPI
    "2025-08-12",  # CPI
    "2025-09-11",  # CPI
    "2025-10-14",  # CPI
    "2025-11-12",  # CPI
    "2025-12-10",  # CPI
    # 2026
    "2026-01-28",  # FOMC
    "2026-03-18",  # FOMC
    "2026-05-06",  # FOMC
    "2026-06-17",  # FOMC
    "2026-07-29",  # FOMC
    "2026-09-16",  # FOMC
    "2026-10-28",  # FOMC
    "2026-12-16",  # FOMC
    "2026-01-14",  # CPI
    "2026-02-11",  # CPI
    "2026-03-11",  # CPI
    "2026-04-14",  # CPI
    "2026-05-12",  # CPI
    "2026-06-10",  # CPI
    "2026-07-14",  # CPI
    "2026-08-12",  # CPI
    "2026-09-11",  # CPI
    "2026-10-14",  # CPI
    "2026-11-10",  # CPI
    "2026-12-10",  # CPI
    # Triple/Quad witching (3rd Friday of Mar, Jun, Sep, Dec)
    "2025-03-21",
    "2025-06-20",
    "2025-09-19",
    "2025-12-19",
    "2026-03-20",
    "2026-06-19",
    "2026-09-18",
    "2026-12-18",
}


def check_snapshot_time(
    now: datetime | None = None,
    safe_window_start: time = time(10, 30),
    safe_window_end: time = time(15, 30),
    exclusion_dates: set[str] | None = None,
) -> str | None:
    """Return a warning string if outside the safe window or on an
    exclusion date, else ``None``.

    Parameters
    ----------
    now:
        The current datetime.  Defaults to ``datetime.now(tz=US/Eastern)``.
    safe_window_start:
        Start of the safe window (US/Eastern).  Default 10:30.
    safe_window_end:
        End of the safe window (US/Eastern).  Default 15:30.
    exclusion_dates:
        Set of date strings (``YYYY-MM-DD``) to warn on.  Uses the
        built-in set if ``None``.

    Returns
    -------
    str or None
        A warning string if the snapshot is outside the safe window or
        on an exclusion date.  ``None`` if everything looks fine.

    Notes
    -----
    This function does NOT block — the caller decides what to do with
    the warning.  The intent is to flag snapshots that may have stale
    or noisy quotes (e.g. pre-market, after-hours, or event days).
    """
    if exclusion_dates is None:
        exclusion_dates = _EXCLUSION_DATES

    eastern = ZoneInfo("US/Eastern")

    if now is None:
        now = datetime.now(tz=eastern)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=eastern)
    else:
        now = now.astimezone(eastern)

    today_str = now.date().isoformat()
    current_time = now.time()

    # Check exclusion date first
    if today_str in exclusion_dates:
        return (
            f"Snapshot on known event day ({today_str}). "
            "Option data may be noisy or stale."
        )

    # Check time window
    if current_time < safe_window_start or current_time > safe_window_end:
        return (
            f"Snapshot at {current_time.strftime('%H:%M')} US/Eastern "
            f"outside safe window "
            f"({safe_window_start.strftime('%H:%M')}-"
            f"{safe_window_end.strftime('%H:%M')}). "
            "Market may not be open or quotes may be stale."
        )

    return None
