from math import exp

import pytest

from arbfree_vol.ingestion.cleaning import RejectionRule, clean_quotes
from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote


def _clean(quote: Quote, **kwargs):
    expiry_slice = ExpirySlice(expiry_time=kwargs.pop("expiry", 0.5), quotes=[quote])
    return clean_quotes(expiry_slice, 100.0, **kwargs)


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan")])
def test_rejects_nonpositive_or_nonfinite_price(price: float) -> None:
    result = _clean(Quote(strike=100, option_type=OptionType.CALL, price=price, bid=1, ask=2))
    assert result.rejected_quotes[0].rule is RejectionRule.INVALID_PRICE


@pytest.mark.parametrize(
    ("bid", "ask"), [(None, 2.0), (1.0, None), (0.0, 2.0), (2.0, 1.0)]
)
def test_rejects_missing_zero_or_crossed_market(bid: float | None, ask: float | None) -> None:
    result = _clean(Quote(strike=100, option_type=OptionType.CALL, price=1.5, bid=bid, ask=ask))
    assert result.rejected_quotes[0].rule is RejectionRule.INVALID_MARKET


def test_rejects_wide_spread() -> None:
    result = _clean(Quote(strike=100, option_type=OptionType.CALL, price=1, bid=0.5, ask=1.5))
    assert result.rejected_quotes[0].rule is RejectionRule.WIDE_SPREAD


def test_rejects_near_expiry() -> None:
    result = _clean(
        Quote(strike=100, option_type=OptionType.CALL, price=2, bid=1.9, ask=2.1),
        expiry=1 / 365,
    )
    assert result.rejected_quotes[0].rule is RejectionRule.NEAR_EXPIRY


def test_uses_discounted_european_call_lower_bound() -> None:
    maturity = 1.0
    rate = 0.05
    dividend = 0.02
    strike = 80.0
    lower = 100 * exp(-dividend * maturity) - strike * exp(-rate * maturity)
    quote = Quote(
        strike=strike,
        option_type=OptionType.CALL,
        price=lower - 0.01,
        bid=lower - 0.02,
        ask=lower,
    )
    result = _clean(quote, expiry=maturity, risk_free=rate, div_yield=dividend)
    assert result.rejected_quotes[0].rule is RejectionRule.PRICE_BOUND


def test_uses_discounted_european_put_lower_bound() -> None:
    maturity = 1.0
    rate = 0.05
    dividend = 0.02
    strike = 120.0
    lower = strike * exp(-rate * maturity) - 100 * exp(-dividend * maturity)
    quote = Quote(
        strike=strike,
        option_type=OptionType.PUT,
        price=lower - 0.01,
        bid=lower - 0.02,
        ask=lower,
    )
    result = _clean(quote, expiry=maturity, risk_free=rate, div_yield=dividend)
    assert result.rejected_quotes[0].rule is RejectionRule.PRICE_BOUND


def test_records_only_first_rejection_reason() -> None:
    quote = Quote(strike=100, option_type=OptionType.CALL, price=-1, bid=2, ask=1)
    result = _clean(quote)
    assert len(result.rejected_quotes) == 1
    assert result.rejected_quotes[0].rule is RejectionRule.INVALID_PRICE


def test_accepts_valid_quote() -> None:
    quote = Quote(strike=100, option_type=OptionType.CALL, price=6, bid=5.9, ask=6.1)
    result = _clean(quote, risk_free=0.05)
    assert result.accepted_quotes == (quote,)
    assert not result.rejected_quotes
