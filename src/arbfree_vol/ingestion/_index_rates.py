"""Shared rate/dividend helpers for the ingestion layer.

Single source of truth for the index representative-ETF mapping, the
per-expiry put-call-parity dividend yield estimators, the ^IRX
risk-free rate fetch, the per-equity trailing dividend-yield fetch, and
the ``(r, q)`` orchestration that both ``ingestion.yahoo`` and
``ingestion.openbb`` use to source rates.  Both ingestion modules
re-import these names so their call sites keep working unchanged, and so
a fix to the estimation logic lands in exactly one place.

This module does not hard-require ``yfinance`` at import time: it is
imported lazily inside the rate helpers so that importing this module
(or the ``openbb`` ingestion module that re-imports its names) does not
require ``yfinance`` at import time — only those helper paths need it at
call time, and a missing installation degrades to the documented
fallbacks (``None`` / ``r=0.05, q=0.0``) instead of raising.
"""

import logging
import math

from arbfree_vol.models.surface import ExpirySlice
from arbfree_vol.models.option import OptionType

_logger = logging.getLogger(__name__)


# Mapping of index symbols to representative ETFs that track the same
# (or very similar) underlying basket.  Used as a FALLBACK when per-expiry
# put-call parity estimation of q fails (e.g., no ATM call/put pair).
# This is the ETF's TRAILING yield, not the index's market-implied forward
# yield — it is an approximation.  Per-expiry put-call parity is preferred.
_INDEX_REPRESENTATIVE: dict[str, str | None] = {
    "^SPX": "SPY",   # SPY tracks S&P 500, same constituents
    "^NDX": "QQQ",   # QQQ tracks Nasdaq-100
    "^DJI": "DIA",   # DIA tracks Dow Jones
    "^RUT": "IWM",   # IWM tracks Russell 2000
    "^VIX": None,    # VIX has no constituents
    # Add more as needed
}


def _estimate_index_dividend_yield(
    slice_: ExpirySlice,
    spot: float,
    r: float,
) -> float | None:
    """Estimate the dividend yield for one expiry slice via put-call parity.

    Uses 3-5 strikes on each side of ATM (6-10 strikes total) to solve
    the put-call parity relation for q:

        C - P + K * e^{-rT} = S * e^{-qT}
        q = -log((C - P + K * e^{-rT}) / S) / T

    The wider band (vs. just the 3 nearest strikes) averages out
    single-strike quote noise that dominates the <0.10y bucket where
    the bid-ask spread is widest.

    Returns the MEDIAN q across all usable ATM pairs, or None if
    estimation fails (no call/put pair, or invalid values).
    """
    from statistics import median
    from math import exp, log

    if slice_.expiry_time <= 0:
        return None

    by_strike: dict[float, dict[OptionType, float]] = {}
    for q in slice_.quotes:
        by_strike.setdefault(q.strike, {})[q.option_type] = q.price

    # 3-5 strikes on each side of ATM (6-10 total) for noise robustness.
    # Strike grid on SPX/SPY is typically $1 or $5 wide, so 3-5 strikes on
    # each side spans ~$6-$50 around ATM depending on spacing. This wider
    # band averages out single-strike quote noise that dominates the
    # <0.10y bucket where the bid-ask spread is widest.
    all_strikes_sorted = sorted(by_strike.keys())
    atm_idx = min(range(len(all_strikes_sorted)), key=lambda i: abs(all_strikes_sorted[i] - spot))
    window = 5  # strikes on each side
    low_idx = max(0, atm_idx - window)
    high_idx = min(len(all_strikes_sorted), atm_idx + window + 1)
    atm_strikes = all_strikes_sorted[low_idx:high_idx]

    qs: list[float] = []
    for K in atm_strikes:
        sides = by_strike[K]
        if OptionType.CALL not in sides or OptionType.PUT not in sides:
            continue
        C = sides[OptionType.CALL]
        P = sides[OptionType.PUT]
        T = slice_.expiry_time
        numerator = C - P + K * exp(-r * T)
        if numerator <= 0 or spot <= 0:
            continue
        q_est = -log(numerator / spot) / T
        # Sanity check: dividend yield should be in a reasonable range
        if -0.5 < q_est < 0.5:
            qs.append(q_est)

    if not qs:
        return None
    return float(median(qs))


def _get_representative_dividend_yield(symbol: str) -> float | None:
    """Fetch the trailing dividend yield from a representative ETF for an
    index symbol.

    This is a FALLBACK used only when per-expiry put-call parity
    estimation of q fails.  The returned value is the ETF's trailing
    yield (e.g., SPY's ~1.3%), not the index's market-implied forward
    yield — it is an approximation.  Per-expiry put-call parity is
    preferred because it uses the actual options data.

    The optional ``yfinance`` dependency is imported LAZILY, inside the
    guarded try/failure boundary and AFTER the representative mapping is
    checked: a symbol with no representative ETF (``^VIX``-style) never
    imports ``yfinance``, and a missing ``yfinance`` installation makes
    this function return ``None`` (like any other fetch failure) instead
    of raising ``ModuleNotFoundError``.

    Returns None only when the yield is genuinely MISSING (no
    representative mapped, field absent, ``None`` or NaN, or a fetch
    failure).  An OBSERVED zero (``dividendYield == 0.0`` present in the
    representative ETF's info) is a real observation and is returned as
    ``0.0``, logged as "observed as zero" — the caller must NOT treat it
    as a missing value and substitute the fallback (aligns with the
    primary paths, commit 5bf429a).
    """
    rep = _INDEX_REPRESENTATIVE.get(symbol)
    if rep is None:
        return None
    try:
        import yfinance as yf
        rep_ticker = yf.Ticker(rep)
        info = rep_ticker.info or {}
        q = info.get("dividendYield")
        if q is not None and isinstance(q, (int, float)):
            q = float(q)
            if math.isnan(q):
                return None
            if q > 0.50:
                q /= 100.0
            if q == 0.0:
                # An observed zero is a real observation, not a missing
                # value: the yield is used as-is, but the provenance is
                # logged so a zero-yield surface is never silent about
                # where q came from.
                _logger.warning(
                    "Dividend yield for %s observed as zero (representative "
                    "ETF %s has dividendYield present as 0.0); using q=0.0 "
                    "as observed",
                    symbol, rep,
                )
            return q
    except Exception:
        _logger.warning(
            "Failed to fetch representative dividend yield for %s via %s",
            symbol, rep, exc_info=True,
        )
    return None


def estimate_index_dividend_yields(
    slices: list[ExpirySlice],
    spot: float,
    r: float,
    symbol: str,
) -> float:
    """Per-expiry put-call-parity q for index slices, with ETF fallback.

    Mutates each slice's ``div_yield`` where parity estimation succeeds and
    returns the surface-level q (median of parity estimates, else the
    representative ETF yield, else 0.0).  Logs mixed-quality provenance.
    """
    from statistics import median as _median

    per_slice_qs: list[float] = []
    parity_slices: list[float] = []
    failed_parity_slices: list[float] = []
    for sl in slices:
        q_est = _estimate_index_dividend_yield(sl, spot, r)
        if q_est is not None:
            sl.div_yield = q_est
            per_slice_qs.append(q_est)
            parity_slices.append(sl.expiry_time)
        else:
            failed_parity_slices.append(sl.expiry_time)

    q = 0.0
    if per_slice_qs:
        q = _median(per_slice_qs)
    else:
        rep_q = _get_representative_dividend_yield(symbol)
        if rep_q is not None:
            q = rep_q
        # else q stays at 0.0 from the initial assignment

    # Visibility only — no value changes.  Report which slices got a
    # genuine per-expiry parity q and which fell back to the
    # surface-level q, and where that surface q came from, so a mixed
    # q-quality surface is never silent.
    if failed_parity_slices and per_slice_qs:
        _logger.warning(
            "Index %s: q quality is MIXED across slices — %d/%d used "
            "per-expiry put-call parity q (%s); %d/%d used the "
            "surface-level q (median of parity estimates, q=%.6f) "
            "(%s)",
            symbol, len(parity_slices), len(slices), parity_slices,
            len(failed_parity_slices), len(slices), q,
            failed_parity_slices,
        )
    elif not per_slice_qs:
        if q != 0.0:
            _logger.warning(
                "Index %s: put-call parity q failed for all %d slices; "
                "surface q from representative ETF yield (q=%.6f)",
                symbol, len(slices), q,
            )
        elif rep_q is not None:
            # The representative yield was PRESENT and observed as
            # zero (the helper itself logs the observed-zero
            # provenance).  This branch must not claim the yield was
            # unavailable — an observed zero is an observation, not
            # a substitution.
            _logger.warning(
                "Index %s: put-call parity q failed for all %d slices; "
                "representative ETF yield observed as zero; surface "
                "q=0.0 as observed",
                symbol, len(slices),
            )
        else:
            _logger.warning(
                "Index %s: put-call parity q failed for all %d slices "
                "and no representative ETF yield available; surface q "
                "hardcoded to 0.0",
                symbol, len(slices),
            )

    return q


def _get_risk_free_rate() -> float | None:
    """Fetch the 13-week Treasury yield (^IRX) as a decimal.

    Returns None if the ticker is unavailable, the value is zero / None,
    or the ``yfinance`` provider is missing.  ``yfinance`` is imported
    lazily (same policy as the dividend-yield helpers above) so importing
    this module never requires the provider.
    """
    try:
        import yfinance as yf

        irx = yf.Ticker("^IRX")
        info = irx.info or {}
        rate = info.get("regularMarketPrice") or info.get("previousClose")
        if rate is not None and isinstance(rate, (int, float)) and rate > 0:
            return rate / 100.0  # percent -> decimal
    except Exception:
        _logger.warning("Failed to fetch risk-free rate from ^IRX", exc_info=True)
    return None


def _get_dividend_yield(ticker) -> float | None:
    """Fetch the dividend yield from a yfinance ticker info.

    yfinance returns it as a fraction (e.g. 0.013 for 1.3%).
    Returns None only when the field is genuinely MISSING (absent,
    ``None`` or NaN).  An observed zero (``dividendYield == 0.0``
    present in the info dict) is a real observation and is returned as
    ``0.0`` — the caller must NOT treat it as a missing value and
    substitute the fallback.
    """
    try:
        info = ticker.info or {}
        q = info.get("dividendYield")
        if q is not None and isinstance(q, (int, float)):
            q = float(q)
            if math.isnan(q):
                return None
            # yfinance sometimes returns percent (1.01 for 1.01%) and
            # sometimes fraction (0.0101).  A yield above 50% is
            # definitely in percent then divide by 100.
            if q > 0.50:
                q /= 100.0
            return q
    except Exception:
        _logger.warning("Failed to fetch dividend yield", exc_info=True)
    return None


def _index_placeholder_q(symbol: str) -> float:
    """Return the index dividend-yield placeholder (0.0).

    Index symbols estimate q per-expiry via put-call parity after the
    slice loop rather than hardcoding q=0.  The q=0.0 here is a
    PLACEHOLDER, not an observation — it must never be logged as an
    observed zero (the pre-fix code hit the ``q == 0.0`` observed-zero
    branch for every index symbol because the placeholder triggered it).
    """
    _logger.warning(
        "Dividend yield for %s starts at the index default q=0.0 "
        "(placeholder); per-expiry put-call parity estimation runs "
        "after the slice loop",
        symbol,
    )
    return 0.0


def _fetch_equity_q(symbol: str, ticker=None) -> float:
    """Fetch the dividend yield for an EQUITY symbol from ticker info.

    Returns ``q`` as a decimal (``> 0.50`` values are treated as percent
    and divided by 100).  Falls back to ``q=0.0`` with a logged warning
    when unavailable; a genuinely observed zero is logged as observed.
    When the caller already holds a yfinance ``Ticker`` for ``symbol``
    (e.g. ``ingestion.yahoo``'s fetch_chain), pass it as ``ticker`` to
    avoid a second provider fetch; otherwise one is created lazily.
    """
    if ticker is None:
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
        except Exception:
            _logger.warning("Failed to fetch dividend yield", exc_info=True)
    q = None
    if ticker is not None:
        q = _get_dividend_yield(ticker)
    if q is None:
        _logger.warning(
            "Dividend yield unavailable for %s (dividendYield missing "
            "from ticker info); substituting q=0.0",
            symbol,
        )
        return 0.0
    if q == 0.0:
        # An observed zero is a real observation, not a substitution:
        # the value is used as-is, but the provenance is logged so a
        # zero-yield surface is never silent about where q came from.
        _logger.warning(
            "Dividend yield for %s observed as zero (dividendYield "
            "present as 0.0 in ticker info); using q=0.0 as observed",
            symbol,
        )
    return q


def fetch_rates(symbol: str, is_index: bool, ticker=None) -> tuple[float, float]:
    """Fetch the risk-free rate (^IRX) and dividend yield for ``symbol``.

    Returns ``(r, q)`` with defaults ``r=0.05, q=0.0`` on failure.  For
    index symbols (``^SPX``, ``^VIX``, ...) ``q`` starts at the 0.0
    placeholder (logged as such, never as an observed zero) — it is
    replaced by per-expiry put-call parity estimation after the slice
    loop.  For equity symbols ``q`` is the ticker's trailing dividend
    yield, with observed-zeros preserved.  ``ticker`` is the caller's
    already-created yfinance ``Ticker`` for ``symbol`` when available
    (avoids a second provider fetch); otherwise one is created lazily.
    """
    r = _get_risk_free_rate()
    if is_index:
        q = _index_placeholder_q(symbol)
    else:
        q = _fetch_equity_q(symbol, ticker)

    if r is None:
        _logger.warning(
            "Risk-free rate unavailable for %s (^IRX fetch failed or "
            "empty); substituting r=0.05",
            symbol,
        )
        r = 0.05

    return r, q
