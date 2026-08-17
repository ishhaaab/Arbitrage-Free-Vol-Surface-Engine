"""Tests for per-expiry dividend yield estimation in yfinance module.

Tests the _estimate_index_dividend_yield and _get_representative_dividend_yield
helpers using synthetic data and mocked yfinance calls.
"""

from datetime import date
from unittest.mock import patch, MagicMock

from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote


# ── _get_dividend_yield tests ────────────────────────────────────────

def test_get_dividend_yield_observed_zero_is_preserved() -> None:
    """An observed ``dividendYield == 0.0`` (the field is PRESENT with
    value zero) must be returned as 0.0 — NOT treated as missing.

    The audit found both ingestion paths treated ``q == 0`` as missing
    (only ``q > 0`` counted as present), so an observed zero was
    silently substituted with the fallback.  A present-zero is a real
    observation and must flow through unchanged."""
    from arbfree_vol.ingestion.yahoo import _get_dividend_yield

    ticker = MagicMock()
    ticker.info = {"regularMarketPrice": 450.0, "dividendYield": 0.0}

    q = _get_dividend_yield(ticker)
    assert q == 0.0


def test_get_dividend_yield_missing_returns_none() -> None:
    """A MISSING dividendYield (field absent, None, or NaN) returns
    None — the caller's fallback path is the only one that may
    substitute q=0.0."""
    from arbfree_vol.ingestion.yahoo import _get_dividend_yield

    # Field absent entirely
    ticker_absent = MagicMock()
    ticker_absent.info = {"regularMarketPrice": 450.0}
    assert _get_dividend_yield(ticker_absent) is None

    # Field present but None
    ticker_none = MagicMock()
    ticker_none.info = {"regularMarketPrice": 450.0, "dividendYield": None}
    assert _get_dividend_yield(ticker_none) is None

    # Field present but NaN (float('nan')) — missing, not zero
    ticker_nan = MagicMock()
    ticker_nan.info = {
        "regularMarketPrice": 450.0, "dividendYield": float("nan"),
    }
    assert _get_dividend_yield(ticker_nan) is None


# ── _estimate_index_dividend_yield tests ─────────────────────────────

def test_estimate_index_dividend_yield_via_parity() -> None:
    """Verify q estimation matches the put-call parity rearrangement."""
    from arbfree_vol.ingestion._index_rates import _estimate_index_dividend_yield
    from arbfree_vol.models.option import OptionContract, BlackScholesInput
    from arbfree_vol.pricing.black_scholes import price

    S, r, T, K = 100.0, 0.05, 1.0, 100.0
    q_true = 0.013  # 1.3% dividend yield

    # Price C and P using Black-Scholes with q_true
    contract_c = OptionContract(
        symbol="X", option_type=OptionType.CALL, strike=K,
        expiry_date=date(2030, 1, 1),
    )
    C = price(BlackScholesInput(
        contract=contract_c, spot=S, expiry_time=T,
        risk_free=r, div_yield=q_true, volatility=0.2,
    ))

    contract_p = OptionContract(
        symbol="X", option_type=OptionType.PUT, strike=K,
        expiry_date=date(2030, 1, 1),
    )
    P = price(BlackScholesInput(
        contract=contract_p, spot=S, expiry_time=T,
        risk_free=r, div_yield=q_true, volatility=0.2,
    ))

    slice_ = ExpirySlice(
        expiry_time=T,
        quotes=[
            Quote(strike=K, option_type=OptionType.CALL, price=C),
            Quote(strike=K, option_type=OptionType.PUT, price=P),
        ],
    )

    q_est = _estimate_index_dividend_yield(slice_, S, r)
    assert q_est is not None
    assert abs(q_est - q_true) < 0.002, (
        f"Estimated q={q_est:.6f} differs from true q={q_true} by > 20bps"
    )


def test_estimate_index_dividend_yield_no_atm_pair() -> None:
    """Slice with only calls → returns None."""
    from arbfree_vol.ingestion._index_rates import _estimate_index_dividend_yield

    slice_ = ExpirySlice(
        expiry_time=1.0,
        quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=5.0),
            Quote(strike=105.0, option_type=OptionType.CALL, price=3.0),
        ],
    )

    q_est = _estimate_index_dividend_yield(slice_, 100.0, 0.05)
    assert q_est is None


def test_estimate_index_dividend_yield_empty_slice() -> None:
    """Empty quotes list → returns None (but ExpirySlice requires min_length=1,
    so we test with a single quote that has no pair)."""
    from arbfree_vol.ingestion._index_rates import _estimate_index_dividend_yield

    slice_ = ExpirySlice(
        expiry_time=1.0,
        quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=5.0),
        ],
    )

    q_est = _estimate_index_dividend_yield(slice_, 100.0, 0.05)
    assert q_est is None


def test_estimate_index_dividend_yield_near_zero_expiry_returns_float() -> None:
    """A near-zero-expiry boundary case with a valid parity pair must
    return a valid float consistent with the true dividend yield.

    The documented contract (_index_rates._estimate_index_dividend_yield,
    "Returns the MEDIAN q across all usable ATM pairs, or None if
    estimation fails") guards only ``expiry_time <= 0``; a genuine
    positive near-zero expiry with a consistent call/put pair is
    estimable — put-call parity recovers q from the tiny C-P gap.
    """
    from arbfree_vol.ingestion._index_rates import _estimate_index_dividend_yield
    from arbfree_vol.models.option import OptionContract, BlackScholesInput
    from arbfree_vol.pricing.black_scholes import price

    S, r, T, K = 100.0, 0.05, 0.001, 100.0
    q_true = 0.013  # 1.3% dividend yield

    contract_c = OptionContract(
        symbol="X", option_type=OptionType.CALL, strike=K,
        expiry_date=date(2030, 1, 1),
    )
    C = price(BlackScholesInput(
        contract=contract_c, spot=S, expiry_time=T,
        risk_free=r, div_yield=q_true, volatility=0.2,
    ))
    contract_p = OptionContract(
        symbol="X", option_type=OptionType.PUT, strike=K,
        expiry_date=date(2030, 1, 1),
    )
    P = price(BlackScholesInput(
        contract=contract_p, spot=S, expiry_time=T,
        risk_free=r, div_yield=q_true, volatility=0.2,
    ))

    slice_ = ExpirySlice(
        expiry_time=T,
        quotes=[
            Quote(strike=K, option_type=OptionType.CALL, price=C),
            Quote(strike=K, option_type=OptionType.PUT, price=P),
        ],
    )

    q_est = _estimate_index_dividend_yield(slice_, S, r)
    assert q_est is not None, (
        "a valid parity pair at near-zero expiry must yield an estimate, "
        "not None (only expiry_time <= 0 or missing pairs return None)"
    )
    assert isinstance(q_est, float)
    assert abs(q_est - q_true) < 0.002, (
        f"Estimated q={q_est:.6f} differs from true q={q_true} by > 20bps"
    )


# ── _get_representative_dividend_yield tests ─────────────────────────

@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
def test_representative_dividend_yield_spx(mock_ticker_class) -> None:
    """^SPX → SPY mapping returns SPY's dividend yield."""
    from arbfree_vol.ingestion._index_rates import _get_representative_dividend_yield

    mock_ticker = MagicMock()
    mock_ticker.info = {"dividendYield": 0.013}
    mock_ticker_class.return_value = mock_ticker

    q = _get_representative_dividend_yield("^SPX")
    assert q is not None
    assert abs(q - 0.013) < 1e-6
    mock_ticker_class.assert_called_once_with("SPY")


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
def test_representative_dividend_yield_vix(mock_ticker_class) -> None:
    """^VIX → None (no representative ETF)."""
    from arbfree_vol.ingestion._index_rates import _get_representative_dividend_yield

    q = _get_representative_dividend_yield("^VIX")
    assert q is None
    mock_ticker_class.assert_not_called()


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
def test_representative_dividend_yield_unknown_symbol(mock_ticker_class) -> None:
    """Unknown index symbol → None."""
    from arbfree_vol.ingestion._index_rates import _get_representative_dividend_yield

    q = _get_representative_dividend_yield("^UNKNOWN")
    assert q is None
    mock_ticker_class.assert_not_called()


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
def test_representative_dividend_yield_handles_percent(mock_ticker_class) -> None:
    """When yfinance returns percent (>0.50), it's converted to fraction."""
    from arbfree_vol.ingestion._index_rates import _get_representative_dividend_yield

    mock_ticker = MagicMock()
    mock_ticker.info = {"dividendYield": 1.3}  # 1.3% as percent
    mock_ticker_class.return_value = mock_ticker

    q = _get_representative_dividend_yield("^SPX")
    assert q is not None
    assert abs(q - 0.013) < 1e-6


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
def test_representative_dividend_yield_handles_exception(mock_ticker_class) -> None:
    """When yfinance raises, returns None gracefully."""
    from arbfree_vol.ingestion._index_rates import _get_representative_dividend_yield

    mock_ticker_class.side_effect = Exception("network error")

    q = _get_representative_dividend_yield("^SPX")
    assert q is None


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
def test_representative_dividend_yield_observed_zero_is_preserved(
    mock_ticker_class, caplog,
) -> None:
    """An OBSERVED zero representative-ETF yield (``dividendYield``
    PRESENT as 0.0 in the representative ticker's info) must be returned
    as ``0.0`` and logged as "observed as zero" — NOT treated as missing.

    Regression (pre-fix): the representative fallback required ``q > 0``,
    so a present-zero was silently treated as missing and the caller fell
    through to the last-resort "no representative ETF yield available;
    hardcoded to 0.0" path.  Mirrors the primary-path semantics from
    commit 5bf429a: present-zero is an observation, absent/None/NaN is
    missing."""
    import logging
    from arbfree_vol.ingestion._index_rates import _get_representative_dividend_yield

    mock_ticker = MagicMock()
    mock_ticker.info = {"dividendYield": 0.0}  # observed zero, present
    mock_ticker_class.return_value = mock_ticker

    with caplog.at_level(logging.WARNING, logger="arbfree_vol.ingestion._index_rates"):
        q = _get_representative_dividend_yield("^SPX")

    assert q == 0.0
    assert "observed as zero" in caplog.text, (
        f"expected the observed-zero warning, got: {caplog.text}"
    )
    assert "substituting" not in caplog.text, (
        f"an observed zero must NOT be logged as a substitution, got: "
        f"{caplog.text}"
    )


@patch("arbfree_vol.ingestion.yahoo.yf.Ticker")
def test_representative_dividend_yield_missing_returns_none(mock_ticker_class) -> None:
    """A MISSING representative-ETF yield (field absent, ``None``, or
    NaN) returns None — the caller's last-resort substitution is the
    only path that may set q=0.0."""
    from arbfree_vol.ingestion._index_rates import _get_representative_dividend_yield

    # Field absent entirely
    mock_ticker_absent = MagicMock()
    mock_ticker_absent.info = {}
    mock_ticker_class.return_value = mock_ticker_absent
    assert _get_representative_dividend_yield("^SPX") is None

    # Field present but None
    mock_ticker_none = MagicMock()
    mock_ticker_none.info = {"dividendYield": None}
    mock_ticker_class.return_value = mock_ticker_none
    assert _get_representative_dividend_yield("^SPX") is None

    # Field present but NaN — missing, not zero
    mock_ticker_nan = MagicMock()
    mock_ticker_nan.info = {"dividendYield": float("nan")}
    mock_ticker_class.return_value = mock_ticker_nan
    assert _get_representative_dividend_yield("^SPX") is None


def test_representative_returns_none_when_yfinance_unavailable(monkeypatch) -> None:
    """A missing ``yfinance`` installation must make the function return
    ``None`` — NOT raise ``ModuleNotFoundError``.

    The lazy ``import yfinance`` sits inside the guarded failure
    boundary, so an import failure degrades exactly like any other fetch
    failure (returns None).  The pre-fix code imported yfinance BEFORE
    the try block, so a missing installation propagated an exception
    through the fallback path.
    """
    import sys

    # Load the parent module first so its top-level yfinance import has
    # already happened; then simulate yfinance being absent for the
    # function's own lazy import.
    from arbfree_vol.ingestion._index_rates import _get_representative_dividend_yield

    monkeypatch.setitem(sys.modules, "yfinance", None)

    q = _get_representative_dividend_yield("^SPX")
    assert q is None


def test_get_risk_free_rate_returns_none_when_yfinance_unavailable(monkeypatch) -> None:
    """A missing ``yfinance`` installation must make the ^IRX fetch return
    ``None`` — NOT raise ``ModuleNotFoundError``.

    Same degradation contract as the dividend-yield helpers: the lazy
    ``import yfinance`` sits inside the guarded failure boundary, so an
    import failure returns None exactly like any other fetch failure.
    """
    import sys

    from arbfree_vol.ingestion._index_rates import _get_risk_free_rate

    monkeypatch.setitem(sys.modules, "yfinance", None)

    r = _get_risk_free_rate()
    assert r is None


def test_representative_no_mapping_does_not_attempt_yfinance_import(monkeypatch) -> None:
    """A symbol with no representative ETF (``^VIX``) must not import
    ``yfinance`` at all: the representative mapping is checked BEFORE
    the lazy import, so a ``^VIX``-style symbol never touches the
    optional dependency."""
    import builtins

    calls = {"yfinance_imports": 0}
    real_import = builtins.__import__

    def _recording_import(name, *args, **kwargs):
        if name == "yfinance" or name.startswith("yfinance."):
            calls["yfinance_imports"] += 1
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _recording_import)

    from arbfree_vol.ingestion._index_rates import _get_representative_dividend_yield

    q = _get_representative_dividend_yield("^VIX")
    assert q is None
    assert calls["yfinance_imports"] == 0, (
        "a symbol with no representative ETF must not import yfinance; "
        "the mapping check must run before the lazy import"
    )
