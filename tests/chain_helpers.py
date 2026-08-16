"""Shared chain/surface builders for the option-chain and arbitrage tests.

The same tiny builders — a minimal option-chain DataFrame, a Black-Scholes
price for a single contract, and a one-slice ``VolSurface`` wrapper — were
duplicated across several test modules with drifted constants.  They live
here so every consumer prices from the same spot/rate/expiry conventions.
"""

from datetime import date

import pandas as pd

from arbfree_vol.models.option import (
    BlackScholesInput,
    OptionContract,
    OptionType,
)
from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface
from arbfree_vol.pricing.black_scholes import price

SPOT = 100.0
RISK_FREE = 0.05
DIV_YIELD = 0.0
T = 1.0
_DUMMY_DATE = date(2030, 1, 1)


def _bs_price(option_type: OptionType, strike: float, sigma: float = 0.2) -> float:
    contract = OptionContract(
        symbol="X",
        option_type=option_type,
        strike=strike,
        expiry_date=_DUMMY_DATE,
    )
    model = BlackScholesInput(
        contract=contract,
        spot=SPOT,
        expiry_time=T,
        risk_free=RISK_FREE,
        div_yield=DIV_YIELD,
        volatility=sigma,
    )
    return price(model)


def _make_chain_df(strikes, oi, volume, bid, ask):
    """Build a minimal option chain DataFrame (yfinance-style columns)."""
    return pd.DataFrame({
        "strike": strikes,
        "openInterest": oi,
        "volume": volume,
        "bid": bid,
        "ask": ask,
    })


def _surface(quotes: list[Quote]) -> VolSurface:
    """Wrap quotes in a single-slice surface at the shared spot/rate/T."""
    return VolSurface(
        spot=SPOT,
        risk_free=RISK_FREE,
        div_yield=DIV_YIELD,
        slices=[ExpirySlice(expiry_time=T, quotes=quotes)],
    )
