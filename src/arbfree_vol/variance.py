"""Per-slice implied total variance; its shared by arbitrage detection and SVI fitting."""

from datetime import date

from arbfree_vol.models.option import ImpliedVolInput, OptionContract
from arbfree_vol.models.surface import ExpirySlice, VolSurface, get_r, get_q
from arbfree_vol.pricing.implied_vol import implied_vol


def slice_total_variance(surface: VolSurface, s: ExpirySlice) -> dict[float, float]:
    """Maps each quoted strike in the slice to its total variance w = sigma**2 * T.

    Quotes whose price admits no implied vol (arb-violating) are dropped.
    """
    out: dict[float, float] = {}

    for q in s.quotes:
        iv_input = ImpliedVolInput(
            contract=OptionContract(
                symbol="_",  # placeholder symbol; not used in any calc
                option_type=q.option_type,
                strike=q.strike,
                expiry_date=date(2004, 1, 1),  # placeholder date; not used in any calc
            ),
            spot=surface.spot,
            expiry_time=s.expiry_time,
            risk_free=get_r(surface, s),
            div_yield=get_q(surface, s),
            market_price=q.price,
        )

        sigma = implied_vol(iv_input)
        if sigma is not None:
            out[q.strike] = sigma**2 * s.expiry_time

    return out
