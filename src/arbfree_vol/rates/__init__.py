"""Rate curves: pillars, interpolation, and FRED fetch.

Primary names
-------------
* :class:`arbfree_vol.rates.curve.YieldTermStructure` — zero curve
  (``zero_rate(T)`` / ``discount(T)`` / ``forward_rate(t1,t2)``).
* :class:`arbfree_vol.rates.curve.Pillar`
* :func:`arbfree_vol.rates.fred.build_fred_curve` — FRED Treasury+SOFR
  curve with disk cache and flat fallback.
* :func:`arbfree_vol.rates.fred.fetch_treasury_curve` — raw pillars.

The ingestion layer now builds a curve from FRED (with ``^IRX`` as a
secondary fallback) and threads ``r(T)`` per slice via
``ExpirySlice.risk_free`` / ``get_r`` — single-rate call sites keep
working because a flat curve's ``zero_rate(T)`` is constant.
"""

from arbfree_vol.rates.curve import YieldTermStructure, Pillar
from arbfree_vol.rates.fred import (
    FRED_SOFR_SERIES,
    FRED_TREASURY_SERIES,
    SOFR_T,
    build_fred_curve,
    fetch_treasury_curve,
)

__all__ = [
    "YieldTermStructure",
    "Pillar",
    "build_fred_curve",
    "fetch_treasury_curve",
    "FRED_TREASURY_SERIES",
    "FRED_SOFR_SERIES",
    "SOFR_T",
]
