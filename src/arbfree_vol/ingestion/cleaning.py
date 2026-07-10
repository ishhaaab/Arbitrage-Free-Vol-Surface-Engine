"""Quote cleaning: reject bad quotes with an audit trail.

Each rule can be enabled/disabled and configured.  Rejected quotes are
preserved with the reason for auditability.
"""

from arbfree_vol.models.surface import Quote, ExpirySlice
from arbfree_vol.models.option import OptionType

from enum import Enum
from math import log
from dataclasses import dataclass



class RejectionRule(str, Enum):
    NEGATIVE_PRICE=  "negative_price"
    ZERO_PRICE=  "zero_price"
    ZERO_BID_OR_ASK=  "zero_bid_or_ask"
    CROSSED_MARKET=  "crossed_market"
    WIDE_SPREAD=  "wide_spread"
    NEAR_EXPIRY=  "near_expiry"
    INTRINSIC_VIOLATION=  "intrinsic_violation"
    DEEP_MONEYNESS=  "deep_moneyness"


@dataclass(frozen=True, slots=True)
class RejectionRecord:
    """Audit record for a quote that was filtered out by a cleaning rule."""
    quote: Quote
    rule: RejectionRule
    detail: str


def _check_zero_price(q: Quote) -> RejectionRecord | None:
    """Reject if price is exactly zero (IV solver will fail)."""
    if q.price== 0:
        return RejectionRecord(q, RejectionRule.ZERO_PRICE, f"price=0")
    return None


def _check_negative_price(q: Quote) -> RejectionRecord | None:
    """Reject if the mid, bid, or ask is negative."""
    if q.price is not None and q.price < 0:
        return RejectionRecord(q, RejectionRule.NEGATIVE_PRICE, f"price={q.price}")
    
    if q.bid is not None and q.bid < 0:
        return RejectionRecord(q, RejectionRule.NEGATIVE_PRICE, f"bid={q.bid}")
    
    if q.ask is not None and q.ask < 0:
        return RejectionRecord(q, RejectionRule.NEGATIVE_PRICE, f"ask={q.ask}")
    return None


def _check_zero_bid_or_ask(q: Quote) -> RejectionRecord | None:
    """Reject if bid/ask is zero or missing AND we are checking spread."""


    if q.bid is None or q.ask is None:
        return None  # missing data is OK if we only check price
    
    if q.bid==0 or q.ask== 0:
        return RejectionRecord(q, RejectionRule.ZERO_BID_OR_ASK, f"bid={q.bid}, ask={q.ask}")
    return None


def _check_crossed_market(q: Quote) -> RejectionRecord | None:
    """Reject if bid > ask (market data error)."""
    if q.bid is None or q.ask is None:
        return None
    
    if q.bid > q.ask:
        return RejectionRecord(q, 
                               RejectionRule.CROSSED_MARKET, 
                               f"bid={q.bid} > ask={q.ask}")
    return None


def _check_wide_spread(q: Quote, max_ratio: float=  0.5) -> RejectionRecord | None:
    """Reject if relative spread (ask-bid)/mid exceeds max_ratio."""
    if q.bid is None or q.ask is None:
        return None
    
    if q.bid <= 0 or q.ask <= 0:
        return None
    
    mid=  (q.bid + q.ask) / 2.0
    if mid <= 0:
        return None
    
    ratio=  (q.ask - q.bid) / mid
    if ratio > max_ratio:
        return RejectionRecord(q, 
                               RejectionRule.WIDE_SPREAD, 
                               f"spread/mid={ratio:.4f} > {max_ratio}")
    return None


def _check_near_expiry(sl: ExpirySlice, q: Quote, min_T: float) -> RejectionRecord | None:
    """Reject if expiry is too close (IV inversion unstable)."""
    if sl.expiry_time < min_T:
        return RejectionRecord(q, 
                               RejectionRule.NEAR_EXPIRY, 
                               f"T={sl.expiry_time:.4f} < {min_T}")
    return None


def _check_intrinsic_violation(sl: ExpirySlice, 
                               q: Quote, 
                               spot: float) -> RejectionRecord | None:
    
    """Reject if price is below intrinsic value bound.

    For a call: price >= max(0, S - K) * e^(-qT)  (approx using spot directly)
    For a put:  price >= max(0, K - S) * e^(-rT)
    """
    intrinsic_call=  max(0.0, spot - q.strike)
    intrinsic_put=  max(0.0, q.strike - spot)

    if q.option_type== OptionType.CALL and q.price < intrinsic_call - 1e-6:
        return RejectionRecord(q, 
                               RejectionRule.INTRINSIC_VIOLATION,
                               f"call price={q.price:.4f} < intrinsic={intrinsic_call:.4f}")
    

    if q.option_type== OptionType.PUT and q.price < intrinsic_put - 1e-6:
        return RejectionRecord(q, 
                               RejectionRule.INTRINSIC_VIOLATION,
                               f"put price={q.price:.4f} < intrinsic={intrinsic_put:.4f}")
    return None


def _check_deep_moneyness(sl: ExpirySlice, 
                          q: Quote, 
                          spot: float, 
                          max_k: float=  1.5) -> RejectionRecord | None:
    """Reject if log moneyness is out of [-max_k, max_k].

    Approximate forward using spot directly by ignoring r/q for the
    cleaning step (the precise value is computed later in fwd_curve).
    """
    if spot <= 0 or q.strike <= 0:
        return None
    k=  log(q.strike / spot)
    if abs(k) > max_k:
        return RejectionRecord(q, 
                               RejectionRule.DEEP_MONEYNESS, 
                               f"|k|={abs(k):.4f} > {max_k}")
    return None


def clean_quotes(
        sl: ExpirySlice,
        spot: float,
        min_T: float=  7.0 / 365.0,
        max_spread_ratio: float=  0.5,
        max_log_moneyness: float=  1.5) -> tuple[list[Quote], list[RejectionRecord]]:
    
    """Apply all cleaning rules to a slice.

    Returns (kept_quotes, rejected_records). The first rejection
    encountered for a quote is recorded & subsequent rules are not
    evaluated for that quote.
    """
    kept: list[Quote]=  []
    rejected: list[RejectionRecord]=  []

    for q in sl.quotes:
        checkers=  [
            lambda q= q: _check_negative_price(q),
            lambda q= q: _check_zero_price(q),
            lambda q= q: _check_zero_bid_or_ask(q),
            lambda q= q: _check_crossed_market(q),
            lambda q= q: _check_wide_spread(q, max_spread_ratio),
            lambda q= q: _check_near_expiry(sl, q, min_T),
            lambda q= q: _check_intrinsic_violation(sl, q, spot),
            lambda q= q: _check_deep_moneyness(sl, q, spot, max_log_moneyness),
        ]
        rejected_q: RejectionRecord | None=  None

        for check in checkers:
            result=  check()
            if result is not None:
                rejected_q=  result
                break
            
        if rejected_q is not None:
            rejected.append(rejected_q)
        else:
            kept.append(q)

    return kept, rejected
