"""Unit tests for ``arbfree_vol.data.audit`` (promoted audit library).

The audit script was split into a testable library + thin driver; this
file is the library's test surface: metric helpers, the offline
fixture-mode audit, the N/A discipline in comparison conclusions, and
the Issue #15 markdown writer (tested against a tmp file, never the
real ``docs/issues.md``).
"""

import pytest

from arbfree_vol.data import audit
from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface


# ---------- helpers ----------


def _chain_df(rows: list[dict]):
    import pandas as pd

    return pd.DataFrame(rows)


def _atm_calls_puts(spot: float = 100.0) -> tuple:
    """Synthetic chains: 5 ATM strikes (98-102) + 1 far strike (110)."""
    calls = _chain_df([
        {"strike": 98, "bid": 1.0, "ask": 1.0, "volume": 0, "openInterest": 10,
         "impliedVolatility": 0.2},
        {"strike": 100, "bid": 1.0, "ask": 2.0, "volume": 100, "openInterest": 20,
         "impliedVolatility": 0.2},
        {"strike": 110, "bid": 0.1, "ask": 0.2, "volume": 999, "openInterest": 99,
         "impliedVolatility": 0.2},
    ])
    puts = _chain_df([
        {"strike": 99, "bid": 2.0, "ask": 4.0, "volume": 200, "openInterest": 5,
         "impliedVolatility": 0.2},
        {"strike": 101, "bid": 3.0, "ask": 6.0, "volume": 300, "openInterest": 30,
         "impliedVolatility": 0.2},
        {"strike": 102, "bid": 4.0, "ask": 8.0, "volume": 400, "openInterest": 40,
         "impliedVolatility": 0.2},
    ])
    return calls, puts


def _slice(T: float) -> ExpirySlice:
    return ExpirySlice(
        expiry_time=T,
        quotes=[Quote(strike=100.0, option_type=OptionType.CALL, price=5.0)],
    )


def _surface(*Ts: float) -> VolSurface:
    return VolSurface(
        spot=100.0,
        risk_free=0.05,
        div_yield=0.01,
        slices=[_slice(T) for T in Ts],
    )


def _result(
    n_fitted: int = 5,
    n_fallback: int = 2,
    drops: int = 3,
    dips: int = 1,
    max_pct: float = 12.5,
) -> dict:
    return {
        "rows": [],
        "fallback_rows": [],
        "ok_rows": [],
        "n_fitted": n_fitted,
        "fallback_slices": [0.1, 0.3],
        "failed_slices": [],
        "n_quality_drops": drops,
        "theta_dips": dips,
        "theta_max_dip_pct": max_pct,
        "tenor_breakdown": {},
    }


# ---------- compute_atm_quality_metrics ----------


def test_atm_quality_metrics_medians_and_zero_counts() -> None:
    calls, puts = _atm_calls_puts()
    m = audit.compute_atm_quality_metrics(calls, puts, spot=100.0)

    # 5 ATM strikes (110 excluded); OI 10,20,5,30,40 -> median 20
    assert m["n_atm_strikes"] == 5
    assert m["median_OI"] == 20.0
    # volume 0,100,200,300,400 -> median 200
    assert m["median_volume"] == 200.0
    # spreads 0, 66.7, 66.7, 66.7, 66.7 -> median 66.7
    assert m["median_bid_ask_pct"] == pytest.approx(66.6666667)
    assert m["zero_vol_count"] == 1
    assert m["zero_oi_count"] == 0
    assert m["zero_quote_count"] == 0


def test_atm_quality_metrics_empty_band_is_zero_not_nan() -> None:
    """No strikes in band: counts are 0, spread median NaN (never a fake 0)."""
    import math

    calls, puts = _atm_calls_puts()
    m = audit.compute_atm_quality_metrics(calls, puts, spot=500.0)
    assert m["n_atm_strikes"] == 0
    assert m["median_OI"] == 0
    assert m["zero_vol_count"] == 0
    assert math.isnan(m["median_bid_ask_pct"])


def test_atm_quality_metrics_zero_quotes_counted() -> None:
    calls = _chain_df([
        {"strike": 100, "bid": 0.0, "ask": 0.0, "volume": 0, "openInterest": 7,
         "impliedVolatility": 0.2},
    ])
    puts = _chain_df([])
    m = audit.compute_atm_quality_metrics(calls, puts, spot=100.0)
    assert m["zero_quote_count"] == 1
    assert m["n_atm_strikes"] == 1


def test_atm_quality_metrics_infinite_quotes_preserved_not_zero() -> None:
    """Infinities behave exactly like pandas ``fillna(0)``: left as-is.

    fillna(0) only replaces NaN.  Rewriting ``inf -> ~1.8e308`` (the
    nan_to_num default) would turn the inf-inf quote pair's NaN spread
    into a fake 0.0 and shrink median_OI from inf to a huge float.
    """
    import numpy as np

    calls = _chain_df([
        {"strike": 100, "bid": np.inf, "ask": np.inf, "volume": 1,
         "openInterest": np.inf, "impliedVolatility": 0.2},
        {"strike": 102, "bid": 10.0, "ask": 10.5, "volume": 2,
         "openInterest": 42, "impliedVolatility": 0.2},
    ])
    puts = _chain_df([])
    m = audit.compute_atm_quality_metrics(calls, puts, spot=100.0)

    # inf survives as inf (a rewritten ~1.8e308 / ~9e307 mean fails this)
    assert np.isinf(m["median_OI"])
    # inf-inf pair contributes NaN (skipped); only the finite pair counts:
    # (10.5 - 10.0) / 10.25 * 100 = 4.8780...
    assert m["median_bid_ask_pct"] == pytest.approx(4.8780487805)
    assert m["zero_oi_count"] == 0
    assert m["zero_quote_count"] == 0
    assert m["n_atm_strikes"] == 2


# ---------- compute_per_expiry_oi_drops ----------


def test_per_expiry_oi_drops_rate() -> None:
    calls = _chain_df([
        {"strike": 98, "bid": 1, "ask": 1, "volume": 0, "openInterest": 5},
        {"strike": 100, "bid": 1, "ask": 1, "volume": 0, "openInterest": 10},
        {"strike": 101, "bid": 1, "ask": 1, "volume": 0, "openInterest": None},
        {"strike": 102, "bid": 1, "ask": 1, "volume": 0, "openInterest": 15},
    ])
    puts = _chain_df([])
    info = audit.compute_per_expiry_oi_drops(calls, puts, spot=100.0, min_oi=10)
    assert info["total_strikes"] == 4
    # OI 5 < 10, and None -> 0 via the fillna(0) substitute, so 0 < 10
    assert info["oi_dropped"] == 2
    assert info["drop_rate"] == pytest.approx(0.5)


def test_per_expiry_oi_drops_empty_band() -> None:
    calls = _chain_df([
        {"strike": 50, "bid": 1, "ask": 1, "volume": 0, "openInterest": 0},
    ])
    puts = _chain_df([])
    info = audit.compute_per_expiry_oi_drops(calls, puts, spot=100.0)
    assert info == {"total_strikes": 0, "oi_dropped": 0, "drop_rate": 0.0}


# ---------- compute_theta_dip_severity ----------


def test_theta_dip_severity_monotonic_is_zero() -> None:
    slices_data = [
        (0.1, [(-0.1, 0.02), (0.0, 0.03), (0.1, 0.04)]),
        (0.2, [(-0.1, 0.04), (0.0, 0.05), (0.1, 0.06)]),
        (0.3, [(-0.1, 0.06), (0.0, 0.07), (0.1, 0.08)]),
    ]
    assert audit.compute_theta_dip_severity(slices_data) == {
        "n_dips": 0, "max_dip_pct": 0.0, "mean_dip_pct": 0.0,
    }


def test_theta_dip_severity_counts_and_sizes_one_dip() -> None:
    """0.05 -> 0.04 is a 20% dip; 0.04 -> 0.06 is not a dip."""
    slices_data = [
        (0.1, [(-0.1, 0.02), (0.0, 0.03), (0.1, 0.04)]),
        (0.2, [(-0.1, 0.02), (0.0, 0.05), (0.1, 0.06)]),
        (0.3, [(-0.1, 0.02), (0.0, 0.04), (0.1, 0.06)]),
        (0.4, [(-0.1, 0.02), (0.0, 0.06), (0.1, 0.08)]),
    ]
    info = audit.compute_theta_dip_severity(slices_data)
    assert info["n_dips"] == 1
    assert info["max_dip_pct"] == pytest.approx(20.0)
    assert info["mean_dip_pct"] == pytest.approx(20.0)


def test_theta_dip_severity_single_slice_is_zero() -> None:
    assert audit.compute_theta_dip_severity([(0.1, [(0.0, 0.03)])]) == {
        "n_dips": 0, "max_dip_pct": 0.0, "mean_dip_pct": 0.0,
    }


# ---------- compute_tenor_bucket_breakdown ----------


def test_tenor_bucket_breakdown_assigns_buckets() -> None:
    surface = _surface(0.05, 0.2, 0.4, 0.6, 1.5, 3.0)
    tb = audit.compute_tenor_bucket_breakdown(surface, fallback_Ts=[0.2, 3.0])

    assert tb["< 0.10y"] == {"fallback": 0, "total": 1}
    assert tb["0.10-0.25y"] == {"fallback": 1, "total": 1}
    assert tb["0.25-0.50y"] == {"fallback": 0, "total": 1}
    assert tb["0.50-1.00y"] == {"fallback": 0, "total": 1}
    assert tb["1.00-2.00y"] == {"fallback": 0, "total": 1}
    assert tb["> 2.00y"] == {"fallback": 1, "total": 1}


@pytest.mark.parametrize(
    ("T", "expected_bucket"),
    [
        (0.05, "< 0.10y"),
        (0.0999, "< 0.10y"),
        # Exact boundaries are strict-< (matches the original audit
        # script): a value ON a boundary lands in the NEXT bucket.
        (0.10, "0.10-0.25y"),
        (0.25, "0.25-0.50y"),
        (0.50, "0.50-1.00y"),
        (1.00, "1.00-2.00y"),
        (2.00, "> 2.00y"),
        (3.0, "> 2.00y"),
    ],
)
def test_tenor_bucket_breakdown_exact_boundaries(
    T: float, expected_bucket: str
) -> None:
    """Boundary maturities pin the strict-< bucket assignment."""
    surface = _surface(T)
    tb = audit.compute_tenor_bucket_breakdown(surface, fallback_Ts=[T])

    assert tb[expected_bucket] == {"fallback": 1, "total": 1}
    for bucket, counts in tb.items():
        if bucket != expected_bucket:
            assert counts == {"fallback": 0, "total": 0}


# ---------- audit_surface (offline fixture mode) ----------


@pytest.mark.slow
def test_audit_surface_fixture_offline_shape() -> None:
    """The full audit runs offline on the saved fixture (ticker=None)."""
    surface = audit.load_spx_fixture()
    result = audit.audit_surface(
        "fixture", surface, quality_drops=[], spot=surface.spot, ticker=None
    )

    assert len(result["rows"]) == len(surface.slices)
    assert all(r["tag"] in {"OK", "FALLBACK", "FAILED"} for r in result["rows"])
    # No ticker -> per-expiry metrics are N/A, never observed zeros
    assert all(r["metrics_available"] is False for r in result["rows"])
    assert all(r["n_atm_strikes"] is None for r in result["rows"])
    assert len(result["fallback_rows"]) == len(result["fallback_slices"])
    assert set(result["fallback_slices"]) <= {sl.expiry_time for sl in surface.slices}
    assert result["n_fitted"] + len(result["fallback_slices"]) + len(
        result["failed_slices"]
    ) >= 1
    assert result["theta_dips"] >= 0
    assert result["theta_mean_dip_pct"] >= 0.0
    total = sum(tb["total"] for tb in result["tenor_breakdown"].values())
    assert total == len(surface.slices)


def test_load_spx_fixture() -> None:
    surface = audit.load_spx_fixture()
    assert surface.spot == pytest.approx(7437.63)
    assert len(surface.slices) >= 5
    assert all(len(sl.quotes) > 0 for sl in surface.slices)


@pytest.mark.slow
def test_print_single_audit_report_smoke(capsys) -> None:
    surface = audit.load_spx_fixture()
    result = audit.audit_surface(
        "fixture", surface, quality_drops=[], spot=surface.spot, ticker=None
    )
    audit.print_single_audit_report("fixture", surface, surface.spot, result)
    out = capsys.readouterr().out
    assert "fixture" in out
    assert "Tenor Bucket Breakdown" in out
    assert "metrics N/A" in out  # fixture mode rows render as N/A


# ---------- N/A discipline in conclusions ----------


def test_dip_comparison_lines_na_discipline() -> None:
    """A source that was not measured is N/A — never a zero comparison."""
    lines = audit.build_dip_comparison_lines({})
    assert any("N/A" in line and "unavailable" in line for line in lines)
    assert any("Install with" in line for line in lines)  # openbb absent
    assert not any("fewer theta dips" in line for line in lines)


def test_dip_comparison_lines_ties_and_winners() -> None:
    tie = {
        "yfinance_SPY": {"raw": {"theta_dips": 2}},
        "yfinance_SPX": {"raw": {"theta_dips": 2}},
        "openbb_SPY": {"raw": {"theta_dips": 2}},
    }
    lines = audit.build_dip_comparison_lines(tie)
    assert any("Both SPX (2) and SPY (2)" in line for line in lines)
    assert any("OpenBB (2) and yfinance (2)" in line for line in lines)

    winner = {
        "yfinance_SPY": {"raw": {"theta_dips": 2}},
        "yfinance_SPX": {"raw": {"theta_dips": 1}},
    }
    lines = audit.build_dip_comparison_lines(winner)
    assert any("SPX has fewer theta dips (1) than SPY (2)" in line for line in lines)


def test_dip_comparison_lines_missing_operand_is_na() -> None:
    """One operand missing -> N/A line, no winner/ties conclusion."""
    partial = {
        "yfinance_SPY": {"raw": {"theta_dips": 2}},
        # yfinance_SPX never measured
    }
    lines = audit.build_dip_comparison_lines(partial)
    assert any("no winner/ties conclusion" in line for line in lines)


# ---------- Issue #15 markdown ----------


def test_build_issues_markdown_rows_and_na() -> None:
    results = {
        "yfinance_SPY": {"raw": _result(), "filtered": _result(n_fallback=1)},
        "yfinance_SPX": {"raw": _result(n_fitted=6, n_fallback=3, dips=0),
                         "filtered": None},  # fixture mode
        "openbb_SPY": None,  # source unavailable
    }
    md = audit.build_issues_markdown(results)

    assert "### Underlying / path comparison" in md
    assert "| yfinance/SPY (raw) | 5 | 2 | 3 | 1 | 12.5% |" in md
    assert "| yfinance/SPX (filtered) | N/A (fixture has no raw chains) |" in md
    assert "| OpenBB/SPY | N/A (source unavailable) | N/A | N/A | N/A | N/A |" in md
    assert "**Key question:**" in md
    assert "OpenBB was not available for comparison" in md


def test_write_findings_replaces_section_only(tmp_path) -> None:
    """The bounded section is replaced; later sections are preserved."""
    target = tmp_path / "issues.md"
    target.write_text(
        "intro\n\n"
        "### Underlying / path comparison\n"
        "OLD TABLE CONTENT\n"
        "\n"
        "### Later section\n"
        "keep me\n",
        encoding="utf-8",
    )
    results = {"yfinance_SPY": {"raw": _result(), "filtered": None}}
    audit.write_findings_to_issues(results, issues_path=target)

    content = target.read_text(encoding="utf-8")
    assert "OLD TABLE CONTENT" not in content
    assert "| yfinance/SPY (raw) | 5 | 2 | 3 | 1 | 12.5% |" in content
    assert "### Later section" in content
    assert "keep me" in content
    assert content.startswith("intro")


def test_write_findings_appends_when_no_marker(tmp_path) -> None:
    target = tmp_path / "issues.md"
    target.write_text("only content\n", encoding="utf-8")
    results = {}
    audit.write_findings_to_issues(results, issues_path=target)

    content = target.read_text(encoding="utf-8")
    assert "only content" in content
    assert "### Underlying / path comparison" in content
    # No measured sources -> every row is N/A
    assert "N/A (source unavailable)" in content