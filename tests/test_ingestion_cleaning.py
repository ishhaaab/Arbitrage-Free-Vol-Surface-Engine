"""Tests for the quote cleaning module."""
from arbfree_vol.models.surface import Quote, ExpirySlice
from arbfree_vol.models.option import OptionType
from arbfree_vol.ingestion.cleaning import (
    RejectionRule,
    clean_quotes,
    _check_negative_price,
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


def test_intrinsic_violation_call_otm() -> None:
    # spot=100, strike=110 (OTM call), price=0.5 -> below intrinsic (0)
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
