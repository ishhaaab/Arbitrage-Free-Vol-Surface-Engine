"""Offline unit tests for the FRED rate fetcher (``rates/fred.py``).

The seam split (pure CSV parser -> one-series fetch -> cache layer ->
pillar composition -> flat fallback) is exercised end-to-end WITHOUT
network: ``urlopen`` is faked, the disk cache is redirected to
``tmp_path``, and the in-memory memo is cleared per test.  Every test
here runs with no internet access and no FRED_OFFLINE coupling.
"""

from __future__ import annotations

import json
import os
import urllib.error
from datetime import date

import pytest

from arbfree_vol.rates import fred
from arbfree_vol.rates.curve import YieldTermStructure


@pytest.fixture(autouse=True)
def _isolated_fred(tmp_path, monkeypatch) -> None:
    """Per-test isolation: clean memo, no env offline, tmp disk cache."""
    fred._MEMO.clear()
    monkeypatch.delenv("FRED_OFFLINE", raising=False)
    monkeypatch.setattr(
        fred, "_cache_path", lambda as_of: str(tmp_path / f"{as_of.isoformat()}.json")
    )


class _FakeResp:
    """Minimal urllib response: context manager yielding bytes."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResp":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


# ---------- parse_fred_csv (pure seam) ----------


def test_parse_fred_csv_latest_row_wins() -> None:
    raw = (
        "DATE,VALUE\n"
        "2024-01-02,4.80\n"
        "2024-01-03,4.82\n"
        "2024-01-04,4.85\n"
    )
    assert fred.parse_fred_csv(raw) == pytest.approx(0.0485)


def test_parse_fred_csv_skips_missing_and_malformed() -> None:
    raw = (
        "DATE,VALUE\n"
        "2024-01-02,4.80\n"
        "2024-01-03,.\n"          # FRED's missing marker
        "2024-01-04,\n"           # empty value
        "2024-01-05,not-a-number\n"
        "2024-01-06,4.90\n"
    )
    assert fred.parse_fred_csv(raw) == pytest.approx(0.049)


def test_parse_fred_csv_keeps_last_parseable_when_tail_missing() -> None:
    raw = "DATE,VALUE\n2024-01-02,4.80\n2024-01-03,.\n2024-01-04,\n"
    # trailing missing rows must NOT clobber the last good observation
    assert fred.parse_fred_csv(raw) == pytest.approx(0.048)


def test_parse_fred_csv_empty_or_header_only_returns_none() -> None:
    assert fred.parse_fred_csv("") is None
    assert fred.parse_fred_csv("DATE,VALUE\n") is None
    assert fred.parse_fred_csv("DATE,VALUE\n2024-01-02,.\n") is None


def test_parse_fred_csv_handles_bom() -> None:
    raw = "\ufeffDATE,VALUE\n2024-01-02,4.85\n"
    assert fred.parse_fred_csv(raw) == pytest.approx(0.0485)


# ---------- _fetch_fred_series (network seam) ----------


def test_fetch_fred_series_uses_csv_endpoint_and_parses(
    monkeypatch,
) -> None:
    seen: dict[str, str] = {}

    def fake_urlopen(url, timeout=8.0):
        seen["url"] = url
        return _FakeResp(b"DATE,VALUE\n2024-01-02,4.85\n")

    monkeypatch.setattr(fred.urllib.request, "urlopen", fake_urlopen)
    assert fred._fetch_fred_series("DGS10") == pytest.approx(0.0485)
    assert "id=DGS10" in seen["url"]


def test_fetch_fred_series_network_error_returns_none_and_logs(
    monkeypatch, caplog,
) -> None:
    def fake_urlopen(url, timeout=8.0):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr(fred.urllib.request, "urlopen", fake_urlopen)
    with caplog.at_level("WARNING", logger="arbfree_vol.rates.fred"):
        assert fred._fetch_fred_series("DGS2") is None
    assert "FRED fetch failed for DGS2" in caplog.text


# ---------- fetch_treasury_curve (orchestrator + cache seam) ----------


def _fake_series_values() -> dict[str, float]:
    """One usable value per series, in DECIMAL (the seam's contract).

    ``_fetch_fred_series`` parses FRED's percent quotes into decimals,
    so callers of the seam see e.g. 0.051 for DGS1.
    """
    return {
        "SOFR": 0.05,
        "DGS1MO": 0.048,
        "DGS3MO": 0.049,
        "DGS6MO": 0.05,
        "DGS1": 0.051,
        "DGS2": 0.052,
        "DGS5": 0.053,
        "DGS10": 0.054,
        "DGS30": 0.055,
    }


def test_fetch_treasury_curve_offline_returns_none_without_network(
    monkeypatch,
) -> None:
    def must_not_fetch(*a, **k):
        raise AssertionError("offline mode must not hit the network")

    monkeypatch.setattr(fred, "_fetch_fred_series", must_not_fetch)
    assert fred.fetch_treasury_curve(as_of=date(2026, 1, 15), offline=True) is None


def test_fetch_treasury_curve_env_offline_returns_none(monkeypatch) -> None:
    def must_not_fetch(*a, **k):
        raise AssertionError("FRED_OFFLINE=1 must not hit the network")

    monkeypatch.setattr(fred, "_fetch_fred_series", must_not_fetch)
    monkeypatch.setenv("FRED_OFFLINE", "1")
    assert fred.fetch_treasury_curve(as_of=date(2026, 1, 15)) is None


def test_fetch_treasury_curve_fetches_and_sorts_pillars(monkeypatch) -> None:
    values = _fake_series_values()
    monkeypatch.setattr(
        fred, "_fetch_fred_series", lambda sid, timeout=8.0: values.get(sid)
    )
    pillars = fred.fetch_treasury_curve(as_of=date(2026, 1, 15))
    assert pillars is not None
    # sorted by maturity; SOFR first at 1/365
    ts = [t for t, _ in pillars]
    assert ts == sorted(ts)
    assert ts[0] == fred.SOFR_T
    assert (1.0, pytest.approx(0.051)) in pillars
    # cache file was written
    assert os.path.exists(fred._cache_path(date(2026, 1, 15)))


def test_fetch_treasury_curve_include_sofr_false_skips_sofr(monkeypatch) -> None:
    values = _fake_series_values()
    called: list[str] = []

    def fake_fetch(sid, timeout=8.0):
        called.append(sid)
        return values.get(sid)

    monkeypatch.setattr(fred, "_fetch_fred_series", fake_fetch)
    pillars = fred.fetch_treasury_curve(as_of=date(2026, 1, 15), include_sofr=False)
    assert pillars is not None
    assert "SOFR" not in called
    assert all(t > fred.SOFR_T for t, _ in pillars)  # first pillar is DGS1MO


def test_fetch_treasury_curve_fresh_cache_hit_skips_network(monkeypatch) -> None:
    as_of = date(2026, 1, 15)
    cached = [(0.25, 0.04), (1.0, 0.05)]
    with open(fred._cache_path(as_of), "w", encoding="utf-8") as f:
        json.dump(cached, f)

    def must_not_fetch(*a, **k):
        raise AssertionError("fresh cache must not hit the network")

    monkeypatch.setattr(fred, "_fetch_fred_series", must_not_fetch)
    assert fred.fetch_treasury_curve(as_of=as_of) == cached


def test_fetch_treasury_curve_stale_cache_refetches_and_rewrites(
    monkeypatch,
) -> None:
    as_of = date(2026, 1, 15)
    path = fred._cache_path(as_of)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([(1.0, 0.03)], f)
    os.utime(path, (0, 0))  # epoch -> way past the 24h TTL

    values = _fake_series_values()
    monkeypatch.setattr(
        fred, "_fetch_fred_series", lambda sid, timeout=8.0: values.get(sid)
    )
    pillars = fred.fetch_treasury_curve(as_of=as_of)
    assert pillars is not None
    assert (1.0, pytest.approx(0.051)) in pillars  # refetched, not stale 0.03
    with open(path, encoding="utf-8") as f:
        reloaded = [tuple(p) for p in json.load(f)]
        assert reloaded == pillars  # cache rewritten


def test_fetch_treasury_curve_corrupt_cache_refetches(monkeypatch) -> None:
    as_of = date(2026, 1, 15)
    with open(fred._cache_path(as_of), "w", encoding="utf-8") as f:
        f.write("{not json!!")

    values = _fake_series_values()
    monkeypatch.setattr(
        fred, "_fetch_fred_series", lambda sid, timeout=8.0: values.get(sid)
    )
    pillars = fred.fetch_treasury_curve(as_of=as_of)
    assert pillars is not None
    assert len(pillars) == 9  # SOFR + 8 DGS series


def test_fetch_treasury_curve_empty_cache_ignored(monkeypatch) -> None:
    as_of = date(2026, 1, 15)
    with open(fred._cache_path(as_of), "w", encoding="utf-8") as f:
        json.dump([], f)  # empty list is not a usable cache

    values = _fake_series_values()
    monkeypatch.setattr(
        fred, "_fetch_fred_series", lambda sid, timeout=8.0: values.get(sid)
    )
    assert fred.fetch_treasury_curve(as_of=as_of) is not None


def test_fetch_treasury_curve_all_series_fail_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(fred, "_fetch_fred_series", lambda sid, timeout=8.0: None)
    assert fred.fetch_treasury_curve(as_of=date(2026, 1, 15)) is None


def test_fetch_treasury_curve_memo_avoids_refetch(monkeypatch) -> None:
    values = _fake_series_values()
    calls = {"n": 0}

    def counting_fetch(sid, timeout=8.0):
        calls["n"] += 1
        return values.get(sid)

    monkeypatch.setattr(fred, "_fetch_fred_series", counting_fetch)
    as_of = date(2026, 1, 15)
    first = fred.fetch_treasury_curve(as_of=as_of)
    second = fred.fetch_treasury_curve(as_of=as_of)
    assert first == second
    assert calls["n"] == 9  # 1 SOFR + 8 DGS, fetched exactly once


# ---------- build_fred_curve (flat fallback seam) ----------


def test_build_fred_curve_offline_flat_fallback() -> None:
    curve = fred.build_fred_curve(
        as_of=date(2026, 1, 15),
        offline=True,
        fallback_rate=0.0425,
        day_count="ACT/360",
    )
    assert isinstance(curve, YieldTermStructure)
    assert curve.zero_rate(1.0) == pytest.approx(0.0425)
    assert curve.day_count == "ACT/360"


def test_build_fred_curve_none_pillars_flat_fallback(monkeypatch) -> None:
    monkeypatch.setattr(fred, "fetch_treasury_curve", lambda **k: None)
    curve = fred.build_fred_curve(as_of=date(2026, 1, 15))
    assert curve.zero_rate(1.0) == pytest.approx(0.05)  # default fallback


def test_build_fred_curve_fetch_raises_flat_fallback(
    monkeypatch, caplog,
) -> None:
    def boom(**k):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(fred, "fetch_treasury_curve", boom)
    with caplog.at_level("WARNING", logger="arbfree_vol.rates.fred"):
        curve = fred.build_fred_curve(as_of=date(2026, 1, 15))
    assert curve.zero_rate(1.0) == pytest.approx(0.05)
    assert "FRED curve build failed" in caplog.text


def test_build_fred_curve_pillars_interpolated(monkeypatch) -> None:
    monkeypatch.setattr(
        fred, "fetch_treasury_curve",
        lambda **k: [(0.25, 0.04), (1.0, 0.05)],
    )
    curve = fred.build_fred_curve(as_of=date(2026, 1, 15), day_count="ACT/365F")
    assert curve.zero_rate(0.25) == pytest.approx(0.04)
    assert curve.zero_rate(1.0) == pytest.approx(0.05)
    # linear on r between pillars: T=0.5 -> 0.04 + (1/3)*0.01
    assert curve.zero_rate(0.5) == pytest.approx(0.04 + 0.01 / 3)