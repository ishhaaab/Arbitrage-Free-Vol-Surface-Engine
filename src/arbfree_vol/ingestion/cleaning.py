"""Small, auditable quote-cleaning policy for European options."""

from dataclasses import dataclass
from enum import Enum
from math import exp, isfinite, log

from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote


class RejectionRule(str, Enum):
    INVALID_PRICE = "invalid_price"
    INVALID_MARKET = "invalid_market"
    WIDE_SPREAD = "wide_spread"
    NEAR_EXPIRY = "near_expiry"
    PRICE_BOUND = "european_price_bound"
    OUTSIDE_DOMAIN = "outside_calibration_domain"


@dataclass(frozen=True, slots=True)
class RejectionRecord:
    quote: Quote
    rule: RejectionRule
    detail: str


@dataclass(frozen=True, slots=True)
class CleanResult:
    accepted_quotes: tuple[Quote, ...]
    rejected_quotes: tuple[RejectionRecord, ...]


def _rejection_reason(
    sl: ExpirySlice,
    quote: Quote,
    spot: float,
    risk_free: float,
    div_yield: float,
    min_expiry: float,
    max_spread_ratio: float,
    max_log_moneyness: float,
) -> RejectionRecord | None:
    if not isfinite(quote.price) or quote.price <= 0:
        return RejectionRecord(quote, RejectionRule.INVALID_PRICE, f"price={quote.price}")

    if (
        quote.bid is None
        or quote.ask is None
        or not isfinite(quote.bid)
        or not isfinite(quote.ask)
        or quote.bid <= 0
        or quote.ask <= 0
        or quote.bid > quote.ask
    ):
        return RejectionRecord(
            quote, RejectionRule.INVALID_MARKET, f"bid={quote.bid}, ask={quote.ask}"
        )

    midpoint = (quote.bid + quote.ask) / 2.0
    relative_spread = (quote.ask - quote.bid) / midpoint
    if relative_spread > max_spread_ratio:
        return RejectionRecord(
            quote,
            RejectionRule.WIDE_SPREAD,
            f"spread/mid={relative_spread:.6g} > {max_spread_ratio}",
        )

    if sl.expiry_time < min_expiry:
        return RejectionRecord(
            quote, RejectionRule.NEAR_EXPIRY, f"T={sl.expiry_time:.6g} < {min_expiry}"
        )

    discounted_spot = spot * exp(-div_yield * sl.expiry_time)
    discounted_strike = quote.strike * exp(-risk_free * sl.expiry_time)
    if quote.option_type is OptionType.CALL:
        lower = max(0.0, discounted_spot - discounted_strike)
        upper = discounted_spot
    else:
        lower = max(0.0, discounted_strike - discounted_spot)
        upper = discounted_strike
    if quote.price < lower - 1e-8 or quote.price > upper + 1e-8:
        return RejectionRecord(
            quote,
            RejectionRule.PRICE_BOUND,
            f"price={quote.price:.6g}, bounds=[{lower:.6g}, {upper:.6g}]",
        )

    log_moneyness = log(quote.strike / spot)
    if abs(log_moneyness) > max_log_moneyness:
        return RejectionRecord(
            quote,
            RejectionRule.OUTSIDE_DOMAIN,
            f"|log(K/S)|={abs(log_moneyness):.6g} > {max_log_moneyness}",
        )
    return None


def clean_quotes(
    sl: ExpirySlice,
    spot: float,
    *,
    risk_free: float = 0.0,
    div_yield: float = 0.0,
    min_expiry: float = 7.0 / 365.0,
    max_spread_ratio: float = 0.5,
    max_log_moneyness: float = 1.5,
) -> CleanResult:
    """Apply the policy once and record exactly one reason per rejection."""
    accepted: list[Quote] = []
    rejected: list[RejectionRecord] = []
    for quote in sl.quotes:
        reason = _rejection_reason(
            sl,
            quote,
            spot,
            risk_free,
            div_yield,
            min_expiry,
            max_spread_ratio,
            max_log_moneyness,
        )
        if reason is None:
            accepted.append(quote)
        else:
            rejected.append(reason)
    return CleanResult(tuple(accepted), tuple(rejected))
