"""Per-slice implied total variance; it is shared by arbitrage detection and SVI fitting."""

import logging
from datetime import date

from arbfree_vol.models.option import ImpliedVolInput, OptionContract
from arbfree_vol.models.surface import ExpirySlice, VolSurface, get_r, get_q
from arbfree_vol.pricing.implied_vol import implied_vol

_logger = logging.getLogger(__name__)


def slice_total_variance(surface: VolSurface, s: ExpirySlice) -> dict[float, float]:
    """Maps each quoted strike in the slice to its total variance w = sigma**2 * T.

    Quotes whose price admits no implied vol (arb-violating) are dropped.
    When enough quotes are dropped that the caller can no longer fit the
    slice, a WARNING is logged with the drop count.

    When both a call and a put survive cleaning at the same strike, the
    per-strike value is the *average* of the two independently-computed
    total variances — both sides should agree under put-call parity, and
    averaging reduces quote noise.
    """
    # accumulate (sum, count) per strike; average at the end so that
    # both call and put contribute when both are present.
    acc: dict[float, list[float]] = {}
    n_dropped = 0

    for q in s.quotes:
        iv_input = ImpliedVolInput(
            contract=OptionContract(
                symbol="_",  # placeholder symbol; not used in any calc
                option_type=q.option_type,
                strike=q.strike,
                expiry_date=date.min,  # sentinel; date-based calculations must fail loudly
            ),
            spot=surface.spot,
            expiry_time=s.expiry_time,
            risk_free=get_r(surface, s),
            div_yield=get_q(surface, s),
            market_price=q.price,
        )

        sigma = implied_vol(iv_input)
        if sigma is not None:
            w = sigma ** 2 * s.expiry_time
            if q.strike not in acc:
                acc[q.strike] = []
            acc[q.strike].append(w)
        else:
            n_dropped += 1

    if n_dropped:
        _logger.warning(
            "slice T=%.4f: %d/%d quotes dropped (no implied vol found)",
            s.expiry_time, n_dropped, len(s.quotes),
        )

    # average call and put when both present
    return {strike: sum(values) / len(values) for strike, values in acc.items()}
