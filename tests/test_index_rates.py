"""Unit tests for the shared rate-source seam (``ingestion._index_rates``).

Covers the three seam functions that both ingestion adapters
(``ingestion.yahoo`` and ``ingestion.openbb``) call instead of inlining
rate policy:

- ``resolve_rate_curve``: supplied curve > FRED curve > None (^IRX)
- ``apply_curve_rates``: per-slice ``r(T)`` threading
- ``resolve_index_q``: index parity-``q`` reconcile with fallback
"""

import pytest

from arbfree_vol.ingestion._index_rates import (
    apply_curve_rates,
    resolve_index_q,
    resolve_rate_curve,
)
from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote
from arbfree_vol.rates import YieldTermStructure


def _curve() -> YieldTermStructure:
    """Two-pillar curve: r(0.25y)=0.05, r(1.0y)=0.10 (linear interp)."""
    return YieldTermStructure.from_pillars([(0.25, 0.05), (1.0, 0.10)])


def _slice(expiry_time: float) -> ExpirySlice:
    return ExpirySlice(
        expiry_time=expiry_time,
        quotes=[Quote(strike=100.0, option_type=OptionType.CALL, price=5.0)],
    )


# ---------- resolve_rate_curve ----------


def test_resolve_rate_curve_supplied_curve_wins(monkeypatch) -> None:
    """A supplied curve wins even when use_fred_curve=True (no FRED fetch)."""
    curve = _curve()

    def _boom():
        raise AssertionError("build_fred_curve must not be called")

    monkeypatch.setattr(
        "arbfree_vol.ingestion._index_rates.build_fred_curve", _boom
    )
    assert resolve_rate_curve(curve, use_fred_curve=True) is curve


def test_resolve_rate_curve_fred_path(monkeypatch) -> None:
    """No supplied curve + use_fred_curve=True builds the FRED curve."""
    fred_curve = _curve()
    monkeypatch.setattr(
        "arbfree_vol.ingestion._index_rates.build_fred_curve",
        lambda: fred_curve,
    )
    assert resolve_rate_curve(None, use_fred_curve=True) is fred_curve


def test_resolve_rate_curve_none_means_irx_path(monkeypatch) -> None:
    """No curve and no FRED -> None signals the shared ^IRX orchestration.

    Also verifies build_fred_curve is NOT consulted on the default path.
    """

    def _boom():
        raise AssertionError("build_fred_curve must not be called")

    monkeypatch.setattr(
        "arbfree_vol.ingestion._index_rates.build_fred_curve", _boom
    )
    assert resolve_rate_curve(None, use_fred_curve=False) is None


# ---------- apply_curve_rates ----------


def test_apply_curve_rates_threads_per_slice_r_of_T() -> None:
    slices = [_slice(0.25), _slice(1.0)]
    apply_curve_rates(slices, _curve())
    assert slices[0].risk_free == pytest.approx(0.05)
    assert slices[1].risk_free == pytest.approx(0.10)


def test_apply_curve_rates_none_curve_is_noop() -> None:
    slices = [_slice(0.25)]
    slices[0].risk_free = 0.05  # pre-existing value survives untouched
    apply_curve_rates(slices, None)
    assert slices[0].risk_free == 0.05


def test_apply_curve_rates_empty_slices_is_noop() -> None:
    apply_curve_rates([], _curve())  # must not raise


# ---------- resolve_index_q ----------


def test_resolve_index_q_index_uses_parity_estimate(monkeypatch) -> None:
    """Index symbol + slices: q re-estimated per-expiry via put-call parity."""
    slices = [_slice(0.25)]
    calls: list[tuple] = []

    def _fake_estimate(slices_, spot, r, symbol):
        calls.append((list(slices_), float(spot), float(r), symbol))
        return 0.013

    monkeypatch.setattr(
        "arbfree_vol.ingestion._index_rates.estimate_index_dividend_yields",
        _fake_estimate,
    )
    q = resolve_index_q(
        slices, spot=100.0, r=0.05, symbol="^SPX", fallback_q=0.0
    )
    assert q == 0.013
    assert calls == [(slices, 100.0, 0.05, "^SPX")]


def test_resolve_index_q_equity_keeps_fallback(monkeypatch) -> None:
    """Equity symbols never run parity estimation; pre-loop q survives."""

    def _boom(*args, **kwargs):
        raise AssertionError(
            "parity estimation must not run for equity symbols"
        )

    monkeypatch.setattr(
        "arbfree_vol.ingestion._index_rates.estimate_index_dividend_yields",
        _boom,
    )
    q = resolve_index_q(
        [_slice(0.25)], spot=100.0, r=0.05, symbol="SPY", fallback_q=0.011
    )
    assert q == 0.011


def test_resolve_index_q_empty_chain_keeps_fallback(monkeypatch) -> None:
    """An index symbol with no slices keeps the pre-loop q unchanged."""

    def _boom(*args, **kwargs):
        raise AssertionError(
            "parity estimation must not run without slices"
        )

    monkeypatch.setattr(
        "arbfree_vol.ingestion._index_rates.estimate_index_dividend_yields",
        _boom,
    )
    q = resolve_index_q([], spot=100.0, r=0.05, symbol="^SPX", fallback_q=0.0)
    assert q == 0.0