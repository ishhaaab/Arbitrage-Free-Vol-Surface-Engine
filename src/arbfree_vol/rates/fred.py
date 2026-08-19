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

Structure (deep module)
-----------------------
``fetch_treasury_curve`` is a thin orchestrator over four small seams,
each testable without the network:

* :func:`parse_fred_csv` — pure CSV -> latest-observation parser.
* ``_fetch_fred_series`` — one series over the wire (network seam).
* ``_cache_load`` / ``_cache_save`` — disk cache with TTL (cache seam).
* ``_fetch_pillars`` — SOFR + Treasury series composition (policy seam).

``build_fred_curve`` stays the flat-fallback wrapper on top.
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


def parse_fred_csv(raw: str) -> float | None:
    """Parse FRED's ``DATE,VALUE`` CSV into the latest observation.

    Pure function — no I/O, no state.  FRED quotes percent (4.85 ->
    0.0485); missing observations are ``.`` (or empty) and are skipped;
    malformed values are skipped.  The LAST parseable row wins, matching
    FRED's "latest observation" semantics.
    """
    last_val: float | None = None
    for line in raw.splitlines()[1:]:
        line = line.strip().lstrip("\ufeff")
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


def _cache_path(as_of: date) -> str:
    # repo root is two levels above src/arbfree_vol/rates
    here = os.path.dirname(__file__)
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    d = os.path.join(repo, ".cache", "rates")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{as_of.isoformat()}.json")


def _cache_load(path: str, ttl_s: float) -> list[tuple[float, float]] | None:
    """Cache seam: fresh pillars from disk, or ``None`` to refetch.

    Missing, stale (mtime older than ``ttl_s``), corrupt, or empty cache
    files all yield ``None`` — the caller refetches.  Never raises.
    """
    if not os.path.exists(path):
        return None
    try:
        if time.time() - os.path.getmtime(path) >= ttl_s:
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not (isinstance(data, list) and data):
            return None
        return [(float(t), float(r)) for t, r in data]
    except Exception:
        return None  # corrupt cache — refetch


def _cache_save(path: str, pillars: list[tuple[float, float]]) -> None:
    """Cache seam: best-effort disk write (never raises)."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pillars, f)
    except Exception:
        pass  # cache is an optimisation, not a contract


def _fetch_fred_series(series_id: str, timeout: float = 8.0) -> float | None:
    """Network seam: latest observation for one FRED series via CSV."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        _logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return None
    return parse_fred_csv(raw)


def _fetch_pillars(
    *,
    include_sofr: bool,
    timeout: float,
) -> list[tuple[float, float]] | None:
    """Policy seam: compose the SOFR + Treasury pillar list.

    Returns ``None`` when no series produced a usable observation.
    """
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
    return pillars


def fetch_treasury_curve(
    as_of: date | None = None,
    *,
    include_sofr: bool = True,
    timeout: float = 8.0,
    offline: bool = False,
) -> list[tuple[float, float]] | None:
    """Fetch a Treasury zero-curve as ``[(T, r), ...]``.

    Orchestrator: offline check -> in-memory memo -> disk cache (TTL)
    -> network.  Returns ``None`` on failure / offline so callers can
    fall back.  Results are cached on disk per *as_of* date.
    """
    if as_of is None:
        as_of = date.today()
    if offline or os.environ.get("FRED_OFFLINE") == "1":
        _logger.info("FRED offline mode — skipping fetch")
        return None

    cache_key = f"{as_of.isoformat()}:{include_sofr}"
    if cache_key in _MEMO:
        return _MEMO[cache_key]

    cached = _cache_load(_cache_path(as_of), _CACHE_TTL_S)
    if cached is not None:
        _MEMO[cache_key] = cached
        return cached

    pillars = _fetch_pillars(include_sofr=include_sofr, timeout=timeout)
    if pillars is None:
        return None

    _MEMO[cache_key] = pillars
    _cache_save(_cache_path(as_of), pillars)
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