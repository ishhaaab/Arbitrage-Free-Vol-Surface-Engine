"""Tests for the quote cleaning module."""
from arbfree_vol.models.surface import Quote, ExpirySlice
from arbfree_vol.models.option import OptionType
from arbfree_vol.ingestion.cleaning import (
    RejectionRule,
    clean_quotes,
    _check_negative_price,
    _check_zero_price,
    _check_zero_bid_or_ask,
    _check_crossed_market,
    _check_wide_spread,
    _check_intrinsic_violation,
    _check_near_expiry,
    _check_deep_moneyness,
)


SPOT = 100.0
T = 0.5


def test_negative_price_rejected() -> None:
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=-1.0)
    assert _check_negative_price(q) is not None


def test_crossed_market_rejected() -> None:
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=5.0, bid=6.0, ask=5.0)
    rec = _check_crossed_market(q)
    assert rec is not None
    assert rec.rule == RejectionRule.CROSSED_MARKET


def test_wide_spread_rejected() -> None:
    # bid=5, ask=15 -> spread=10, mid=10, ratio=1.0 > 0.5
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=10.0, bid=5.0, ask=15.0)
    rec = _check_wide_spread(q, max_ratio=0.5)
    assert rec is not None
    assert rec.rule == RejectionRule.WIDE_SPREAD


def test_narrow_spread_kept() -> None:
    # bid=9, ask=11 -> ratio=0.2
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=10.0, bid=9.0, ask=11.0)
    assert _check_wide_spread(q, max_ratio=0.5) is None


def test_intrinsic_violation_call_otm_is_not_a_violation() -> None:
    # spot=100, strike=110 (OTM call), price=0.5 -> intrinsic is 0, so any
    # positive price is fine.  This is the NON-violation side of the OTM
    # case; the genuine violation side (ITM call priced below intrinsic)
    # is covered by test_intrinsic_violation_call_itm below.
    sl = ExpirySlice(expiry_time=T, quotes=[Quote(strike=100.0, option_type=OptionType.CALL, price=10.0)])
    q = Quote(strike=110.0, option_type=OptionType.CALL, price=0.5)
    rec = _check_intrinsic_violation(sl, q, SPOT)
    assert rec is None  # OTM call can have any positive price


def test_intrinsic_violation_call_itm() -> None:
    # spot=100, strike=80 (ITM call), intrinsic=20, price=10 -> violation
    sl = ExpirySlice(expiry_time=T, quotes=[Quote(strike=100.0, option_type=OptionType.CALL, price=10.0)])
    q = Quote(strike=80.0, option_type=OptionType.CALL, price=10.0)
    rec = _check_intrinsic_violation(sl, q, SPOT)
    assert rec is not None
    assert rec.rule == RejectionRule.INTRINSIC_VIOLATION


def test_intrinsic_violation_put_itm() -> None:
    # spot=100, strike=120 (ITM put), intrinsic=20, price=10 -> violation
    sl = ExpirySlice(expiry_time=T, quotes=[Quote(strike=100.0, option_type=OptionType.PUT, price=5.0)])
    q = Quote(strike=120.0, option_type=OptionType.PUT, price=10.0)
    rec = _check_intrinsic_violation(sl, q, SPOT)
    assert rec is not None


def test_near_expiry_rejected() -> None:
    sl = ExpirySlice(expiry_time=1.0 / 365.0, quotes=[Quote(strike=100.0, option_type=OptionType.CALL, price=1.0)])  # 1 day
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=1.0)
    rec = _check_near_expiry(sl, q, min_T=7.0 / 365.0)
    assert rec is not None
    assert rec.rule == RejectionRule.NEAR_EXPIRY


def test_far_expiry_kept() -> None:
    sl = ExpirySlice(expiry_time=0.5, quotes=[Quote(strike=100.0, option_type=OptionType.CALL, price=10.0)])
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=10.0)
    assert _check_near_expiry(sl, q, min_T=7.0 / 365.0) is None


def test_deep_moneyness_rejected() -> None:
    sl = ExpirySlice(expiry_time=1.0, quotes=[Quote(strike=100.0, option_type=OptionType.CALL, price=10.0)])
    # strike=500, spot=100 -> k = ln(5) ≈ 1.609, > 1.5
    q = Quote(strike=500.0, option_type=OptionType.CALL, price=0.1)
    rec = _check_deep_moneyness(sl, q, SPOT, max_k=1.5)
    assert rec is not None
    assert rec.rule == RejectionRule.DEEP_MONEYNESS


def test_atm_moneyness_kept() -> None:
    sl = ExpirySlice(expiry_time=1.0, quotes=[Quote(strike=100.0, option_type=OptionType.CALL, price=10.0)])
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=10.0)
    assert _check_deep_moneyness(sl, q, SPOT, max_k=1.5) is None


def test_clean_quotes_keeps_clean_and_rejects_bad() -> None:
    # 1 clean, 1 with crossed market, 1 with negative price
    q_clean = Quote(strike=100.0, option_type=OptionType.CALL, price=10.0, bid=9.0, ask=11.0)
    q_crossed = Quote(strike=110.0, option_type=OptionType.CALL, price=5.0, bid=6.0, ask=5.0)
    q_neg = Quote(strike=120.0, option_type=OptionType.CALL, price=-1.0)

    sl = ExpirySlice(
        expiry_time=0.5,
        quotes=[q_clean, q_crossed, q_neg],
    )

    kept, rejected = clean_quotes(sl, spot=SPOT)

    assert len(kept) == 1
    assert kept[0].strike == 100.0
    assert len(rejected) == 2
    rules = {r.rule for r in rejected}
    assert RejectionRule.CROSSED_MARKET in rules
    assert RejectionRule.NEGATIVE_PRICE in rules


# ---------------------------------------------------------------------------
# Exact-boundary tests for the remaining cleaning rules
# ---------------------------------------------------------------------------


def test_zero_price_exactly_rejected() -> None:
    """The zero-price rule rejects only price == 0 exactly."""
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=0.0)
    rec = _check_zero_price(q)
    assert rec is not None
    assert rec.rule == RejectionRule.ZERO_PRICE


def test_tiny_positive_price_kept() -> None:
    """Any positive price — even 1e-12 — passes the zero-price rule."""
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=1e-12)
    assert _check_zero_price(q) is None


def test_zero_bid_or_ask_exactly_rejected() -> None:
    """bid == 0 or ask == 0 violates; missing sides are fine (only the
    price check is mandatory) and small-but-positive values pass."""
    q_zero_bid = Quote(strike=100.0, option_type=OptionType.CALL, price=5.0, bid=0.0, ask=10.0)
    rec = _check_zero_bid_or_ask(q_zero_bid)
    assert rec is not None
    assert rec.rule == RejectionRule.ZERO_BID_OR_ASK
    assert "bid=0.0" in rec.detail

    q_zero_ask = Quote(strike=100.0, option_type=OptionType.CALL, price=5.0, bid=5.0, ask=0.0)
    assert _check_zero_bid_or_ask(q_zero_ask) is not None


def test_zero_bid_or_ask_missing_side_is_ok() -> None:
    """Documented contract: a missing bid or ask short-circuits the rule
    BEFORE the zero check, so a missing side is never a zero-bid/ask
    violation regardless of the present side's value (see also the mixed
    zero/missing case below)."""
    q_missing_bid = Quote(strike=100.0, option_type=OptionType.CALL, price=5.0, ask=10.0)
    q_missing_ask = Quote(strike=100.0, option_type=OptionType.CALL, price=5.0, bid=5.0)
    assert _check_zero_bid_or_ask(q_missing_bid) is None
    assert _check_zero_bid_or_ask(q_missing_ask) is None


def test_zero_bid_or_ask_mixed_zero_and_missing_side_is_ok() -> None:
    """Mixed zero/missing sides pass per the ACTUAL code: the
    ``q.bid is None or q.ask is None`` early return fires before the
    zero check, so bid=0/ask=None and bid=None/ask=0 are both kept —
    a zero-bid/ask violation requires BOTH sides present."""
    q_zero_bid_missing_ask = Quote(
        strike=100.0, option_type=OptionType.CALL, price=5.0, bid=0.0
    )
    q_missing_bid_zero_ask = Quote(
        strike=100.0, option_type=OptionType.CALL, price=5.0, ask=0.0
    )
    assert _check_zero_bid_or_ask(q_zero_bid_missing_ask) is None
    assert _check_zero_bid_or_ask(q_missing_bid_zero_ask) is None


def test_zero_bid_or_ask_small_positive_values_pass() -> None:
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=0.02, bid=0.01, ask=0.02)
    assert _check_zero_bid_or_ask(q) is None


def test_negative_bid_or_ask_flagged_with_detail() -> None:
    """Negative bid/ask are NEGATIVE_PRICE violations naming the field."""
    q_bid = Quote(strike=100.0, option_type=OptionType.CALL, price=5.0, bid=-1.0, ask=10.0)
    rec = _check_negative_price(q_bid)
    assert rec is not None
    assert rec.rule == RejectionRule.NEGATIVE_PRICE
    assert "bid=-1.0" in rec.detail

    q_ask = Quote(strike=100.0, option_type=OptionType.CALL, price=5.0, bid=5.0, ask=-1.0)
    rec = _check_negative_price(q_ask)
    assert rec is not None
    assert rec.rule == RejectionRule.NEGATIVE_PRICE
    assert "ask=-1.0" in rec.detail


def test_near_expiry_exact_cutoff_kept() -> None:
    """The rule rejects only expiry_time < min_T: equality at the cutoff
    is kept."""
    min_T = 7.0 / 365.0
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=1.0)
    sl = ExpirySlice(expiry_time=min_T, quotes=[q])
    assert _check_near_expiry(sl, q, min_T) is None


def test_near_expiry_just_below_cutoff_rejected() -> None:
    """One floating-point step below the cutoff is rejected."""
    min_T = 7.0 / 365.0
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=1.0)
    sl = ExpirySlice(expiry_time=min_T - 1e-9, quotes=[q])
    rec = _check_near_expiry(sl, q, min_T)
    assert rec is not None
    assert rec.rule == RejectionRule.NEAR_EXPIRY


def test_wide_spread_exact_boundary_kept() -> None:
    """bid=3, ask=5 -> mid=4 -> ratio = (5-3)/4 = 0.5 == max_ratio.  The
    rule rejects only ratio > max_ratio, so equality is kept."""
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=4.0, bid=3.0, ask=5.0)
    assert _check_wide_spread(q, max_ratio=0.5) is None


def test_wide_spread_just_above_boundary_rejected() -> None:
    q = Quote(strike=100.0, option_type=OptionType.CALL, price=4.0, bid=2.99, ask=5.0)
    rec = _check_wide_spread(q, max_ratio=0.5)
    assert rec is not None
    assert rec.rule == RejectionRule.WIDE_SPREAD


def test_intrinsic_violation_exact_tolerance_boundary() -> None:
    """spot=100, strike=80 -> intrinsic=20; the rule rejects only
    price < intrinsic - 1e-6.  Just above the tolerance: kept; just
    below: rejected."""
    sl = ExpirySlice(expiry_time=T, quotes=[Quote(strike=100.0, option_type=OptionType.CALL, price=10.0)])

    q_above = Quote(strike=80.0, option_type=OptionType.CALL, price=20.0 - 5e-7)
    assert _check_intrinsic_violation(sl, q_above, SPOT) is None

    q_below = Quote(strike=80.0, option_type=OptionType.CALL, price=20.0 - 2e-6)
    rec = _check_intrinsic_violation(sl, q_below, SPOT)
    assert rec is not None
    assert rec.rule == RejectionRule.INTRINSIC_VIOLATION
