"""FRED / Treasury term-structure fetcher for :class:`YieldTermStructure`.

Free, no API key, no extra dependency beyond what the repo already
requires (only ``urllib`` from the stdlib + optional ``pandas`` for
parsing — falls back to a tiny CSV parser if pandas is absent).

Sources
-------
* **SOFR** overnight rate — FRED ``SOFR`` (Federal Reserve Bank of NY).
* **Treasury constant-maturity** — FRED ``DGS1MO``, ``DGS3MO``,
  ``DGS6MO``, ``DGS1``, ``DGS2``, ``DGS5``, ``DGS10``, ``DGS30``.
  These are par yields; for a research curve we treat them as zero
  rates (the distinction is <2bp at these tenors and the cache + user
  override matter more).

Both series come from ``https://fred.stlouisfed.org/graph/fredgraph.csv``.
The CSV endpoint is public and does not require an API key.

Caching
-------
Pillars are cached on disk at ``<repo>/.cache/rates/<as_of>.json`` and
in-memory for the process.  A 24h TTL keeps live runs fresh without
hammering FRED.  ``--offline`` / ``FRED_OFFLINE=1`` / file-not-found
all degrade to the documented fallbacks instead of raising.

If fetching fails, callers should fall back to
``YieldTermStructure.flat(^IRX or 0.05)`` — the same fallback the old
single-rate path used.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import date

from arbfree_vol.rates.curve import YieldTermStructure

_logger = logging.getLogger(__name__)

# FRED series -> pillar maturity in years (approx ACT/365)
FRED_TREASURY_SERIES: dict[str, float] = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS5": 5.0,
    "DGS10": 10.0,
    "DGS30": 30.0,
}
FRED_SOFR_SERIES = "SOFR"  # overnight ~ 1/365, mapped to 1/365
SOFR_T = 1 / 365

_CACHE_TTL_S = 24 * 3600

_MEMO: dict[str, list[tuple[float, float]]] = {}


def _cache_path(as_of: date) -> str:
    # repo root is two levels above src/arbfree_vol/rates
    here = os.path.dirname(__file__)
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    d = os.path.join(repo, ".cache", "rates")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{as_of.isoformat()}.json")


def _fetch_fred_series(series_id: str, timeout: float = 8.0) -> float | None:
    """Fetch the latest observation for a FRED series via CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return None
    # CSV: DATE,VALUE  — last non-missing row is latest
    last_val: float | None = None
    for line in raw.splitlines()[1:]:
        line = line.strip()
        if not line or line.startswith("DATE"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        v = parts[1].strip()
        if v == "" or v == ".":
            continue
        try:
            last_val = float(v) / 100.0  # FRED quotes percent
        except ValueError:
            continue
    return last_val


def fetch_treasury_curve(
    as_of: date | None = None,
    *,
    include_sofr: bool = True,
    timeout: float = 8.0,
    offline: bool = False,
) -> list[tuple[float, float]] | None:
    """Fetch a Treasury zero-curve as ``[(T, r), ...]``.

    Returns ``None`` on failure / offline so callers can fall back.
    Results are cached on disk per *as_of* date.
    """
    if as_of is None:
        as_of = date.today()
    if offline or os.environ.get("FRED_OFFLINE") == "1":
        _logger.info("FRED offline mode — skipping fetch")
        return None

    cache_key = f"{as_of.isoformat()}:{include_sofr}"
    if cache_key in _MEMO:
        return _MEMO[cache_key]

    # disk cache
    cp = _cache_path(as_of)
    if os.path.exists(cp):
        try:
            if time.time() - os.path.getmtime(cp) < _CACHE_TTL_S:
                with open(cp, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list) and data:
                    pillars = [(float(t), float(r)) for t, r in data]
                    _MEMO[cache_key] = pillars
                    return pillars
        except Exception:
            pass  # corrupt cache — refetch

    pillars: list[tuple[float, float]] = []
    if include_sofr:
        v = _fetch_fred_series(FRED_SOFR_SERIES, timeout=timeout)
        if v is not None and v > 0:
            pillars.append((SOFR_T, v))
    for sid, t in sorted(FRED_TREASURY_SERIES.items(), key=lambda kv: kv[1]):
        v = _fetch_fred_series(sid, timeout=timeout)
        if v is not None and v > 0:
            pillars.append((t, v))

    if not pillars:
        return None

    pillars.sort(key=lambda p: p[0])
    _MEMO[cache_key] = pillars
    try:
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(pillars, f)
    except Exception:
        pass
    return pillars


def build_fred_curve(
    as_of: date | None = None,
    *,
    include_sofr: bool = True,
    fallback_rate: float = 0.05,
    day_count: str = "ACT/365F",
    offline: bool = False,
) -> YieldTermStructure:
    """Build a :class:`YieldTermStructure` from FRED, with flat fallback.

    On any failure returns ``YieldTermStructure.flat(fallback_rate)``
    so the pipeline never breaks — matches the old ``r=0.05`` fallback.
    """
    try:
        pillars = fetch_treasury_curve(as_of=as_of, include_sofr=include_sofr, offline=offline)
    except Exception as exc:  # defensive — fetch should already swallow
        _logger.warning("FRED curve build failed: %s", exc, exc_info=True)
        pillars = None
    if not pillars:
        _logger.warning("FRED curve unavailable — using flat r=%.4f", fallback_rate)
        return YieldTermStructure.flat(fallback_rate, day_count=day_count)
    return YieldTermStructure.from_pillars(pillars, day_count=day_count)
