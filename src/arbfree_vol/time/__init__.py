"""Time conventions: DayCount and Calendar.

Mirrors the QuantLib split without taking a hard dependency:

* :class:`DayCount` — converts a date pair to a year fraction (ACT/365F,
  ACT/360, 30/360 Bond Basis) and to a business-day T via a Calendar.
* :class:`Calendar` — US NYSE calendar (weekends + NYSE holidays +
  FOMC/CPI awareness delegated to snapshot_guard) for expiry roll.

``days/365.0`` is preserved as the default (ACT/365F) so existing
fixtures and regression values remain byte-identical unless the caller
opts into a different convention.
"""

from __future__ import annotations

from datetime import date, timedelta


class DayCount:
    """Year-fraction conventions.

    Supported names (case-insensitive, ``/``, ``-`` or ``_`` separators
    accepted)::

        ACT/365F, ACT365F, ACT/365  -> Actual/365 Fixed
        ACT/360, ACT360             -> Actual/360
        30/360, 30360, 30-360        -> 30/360 Bond Basis (US)

    The default is ``ACT/365F`` — the repo's historical ``days/365.0``.
    """

    _ALIASES: dict[str, str] = {
        "ACT/365F": "ACT/365F",
        "ACT365F": "ACT/365F",
        "ACT/365": "ACT/365F",
        "ACT365": "ACT/365F",
        "ACT/360": "ACT/360",
        "ACT360": "ACT/360",
        "30/360": "30/360",
        "30360": "30/360",
        "30-360": "30/360",
        "30_360": "30/360",
    }

    def __init__(self, convention: str = "ACT/365F") -> None:
        key = convention.strip().upper().replace("-", "/").replace("_", "/").replace(" ", "")
        # normalise "ACT/365F" stays, "ACT365F" -> "ACT/365F" via alias
        if "/" not in key and key.startswith("ACT"):
            key = "ACT/" + key[3:]
        if key == "ACT/365F" or key == "ACT/365":
            key = "ACT/365F"
        norm = self._ALIASES.get(key)
        if norm is None:
            # also try without slash for 30/360 variants
            norm = self._ALIASES.get(key.replace("/", ""))
        if norm is None:
            raise ValueError(
                f"Unknown DayCount convention {convention!r}. "
                f"Supported: {sorted(set(self._ALIASES.values()))}"
            )
        self.convention = norm

    def year_fraction(self, start: date, end: date) -> float:
        """Year fraction from *start* (exclusive) to *end* (inclusive).

        For ``end <= start`` returns ``0.0`` (expired / same-day).
        """
        if end <= start:
            return 0.0
        if self.convention == "ACT/365F":
            return (end - start).days / 365.0
        if self.convention == "ACT/360":
            return (end - start).days / 360.0
        # 30/360 Bond Basis
        d1, d2 = start.day, end.day
        m1, m2 = start.month, end.month
        y1, y2 = start.year, end.year
        if d1 == 31:
            d1 = 30
        if d2 == 31 and d1 == 30:
            d2 = 30
        days_360 = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
        return days_360 / 360.0

    def __repr__(self) -> str:  # pragma: no cover
        return f"DayCount({self.convention!r})"


class Calendar:
    """Business-day calendar for expiry handling.

    * ``is_business_day(d)`` — Monday-Friday and not a known NYSE holiday
      (New Year's, MLK, Presidents', Good Friday, Memorial, Juneteenth,
      Independence, Labor, Thanksgiving, Christmas — observed rule applied).
    * ``adjust(d, convention)`` — roll to a business day (``following``,
      ``preceding``, ``modified_following``).
    * ``business_days_between(start, end)`` — ACT/252-style count.

    The holiday set is intentionally small and stable — for production
    calendars plug in ``pandas.tseries.holiday.USFederalHolidayCalendar``
    or ``exchange_calendars`` and inject via ``holidays=``.
    """

    def __init__(
        self,
        name: str = "USNYSE",
        holidays: set[date] | None = None,
    ) -> None:
        self.name = name
        self._holidays = holidays if holidays is not None else _nyse_holidays(
            # pre-generate a window that covers typical option expiries
            start_year=date.today().year - 2,
            end_year=date.today().year + 5,
        )

    def is_business_day(self, d: date) -> bool:
        if d.weekday() >= 5:
            return False
        return d not in self._holidays

    def adjust(self, d: date, convention: str = "following") -> date:
        """Roll *d* to a business day."""
        if self.is_business_day(d):
            return d
        conv = convention.lower()
        if conv == "following":
            while not self.is_business_day(d):
                d += timedelta(days=1)
            return d
        if conv == "preceding":
            while not self.is_business_day(d):
                d -= timedelta(days=1)
            return d
        if conv == "modified_following":
            fwd = d
            while not self.is_business_day(fwd):
                fwd += timedelta(days=1)
            # if we crossed month boundary, go preceding instead
            if fwd.month != d.month:
                bwd = d
                while not self.is_business_day(bwd):
                    bwd -= timedelta(days=1)
                return bwd
            return fwd
        raise ValueError(f"Unknown convention {convention!r}")

    def business_days_between(self, start: date, end: date) -> int:
        if end <= start:
            return 0
        n = 0
        cur = start + timedelta(days=1)
        while cur <= end:
            if self.is_business_day(cur):
                n += 1
            cur += timedelta(days=1)
        return n


def _nyse_holidays(start_year: int, end_year: int) -> set[date]:
    """Small NYSE holiday set (observed rule)."""
    out: set[date] = set()
    for y in range(start_year, end_year + 1):
        # New Year's
        out.add(_observed(date(y, 1, 1)))
        # MLK — 3rd Monday Jan
        out.add(_nth_weekday(y, 1, 0, 3))
        # Presidents' — 3rd Monday Feb
        out.add(_nth_weekday(y, 2, 0, 3))
        # Good Friday — computed from Easter
        out.add(_easter(y) - timedelta(days=2))
        # Memorial — last Monday May
        out.add(_last_weekday(y, 5, 0))
        # Juneteenth
        out.add(_observed(date(y, 6, 19)))
        # Independence
        out.add(_observed(date(y, 7, 4)))
        # Labor — 1st Monday Sep
        out.add(_nth_weekday(y, 9, 0, 1))
        # Thanksgiving — 4th Thursday Nov
        out.add(_nth_weekday(y, 11, 3, 4))
        # Christmas
        out.add(_observed(date(y, 12, 25)))
    return out


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    # advance to first weekday
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d + timedelta(days=7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    # last day of month
    if month == 12:
        d = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        d = date(year, month + 1, 1) - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _easter(year: int) -> date:
    # Anonymous Gregorian algorithm
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    L = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * L) // 451
    month = (h + L - 7 * m + 114) // 31
    day = ((h + L - 7 * m + 114) % 31) + 1
    return date(year, month, day)


# Convenience singletons matching QuantLib naming
ACT365F = DayCount("ACT/365F")
ACT360 = DayCount("ACT/360")
THIRTY_360 = DayCount("30/360")
USNYSE = Calendar("USNYSE")
