"""Per-slice implied total variance; its shared by arbitrage detection and SVI fitting."""

from datetime import date

from arbfree_vol.models.option import ImpliedVolInput, OptionContract
from arbfree_vol.models.surface import ExpirySlice, VolSurface
from arbfree_vol.pricing.implied_vol import implied_vol


def slice_total_variance(surface: VolSurface, s: ExpirySlice) -> dict[float, float]:
    """Maps each quoted strike in the slice to its total variance w = sigma**2 * T.

    Quotes whose price admits no implied vol (arbitrage-violating) are dropped.
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
            risk_free=surface.risk_free,
            div_yield=surface.div_yield,
            market_price=q.price,
        )

        sigma = implied_vol(iv_input)
        if sigma is not None:
            out[q.strike] = sigma**2 * s.expiry_time

    return out
