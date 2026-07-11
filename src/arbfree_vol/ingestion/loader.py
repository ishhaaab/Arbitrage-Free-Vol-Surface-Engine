"""Load option chain data from CSV files into a VolSurface.

CSV schema (one row per quote):
    timestamp, underlying, expiry, strike, option_type, bid, ask, price

Surface level fields (spot, risk_free, div_yield) can be supplied via
function arguments or inferred from the data (spot= first row's value
if all rows agree; r, q default to 0.05 and 0.0).
"""

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.ingestion.cleaning import clean_quotes, RejectionRecord


import csv
from datetime import datetime, date
from pathlib import Path



_REQUIRED_FIELDS=  ("strike", "expiry", "option_type", "price")


def _parse_expiry(value: str, as_of: date | None) -> float:
    """Parse an expiry string (YYYY-MM-DD) and return years to expiry.
    If as_of is None, uses today.
    """
    exp=  datetime.strptime(value, "%Y-%m-%d").date()
    ref=  as_of or date.today()
    days=  (exp - ref).days
    return max(0.0, days / 365.0)


def _parse_option_type(value: str) -> OptionType:
    v=  value.strip().lower()
    if v in ("call", "c"):
        return OptionType.CALL
    
    if v in ("put", "p"):
        return OptionType.PUT
    
    raise ValueError(f"Unknown option type: {value!r}")


def _safe_float(value: str | None) -> float | None:
    if value is None or value== "":
        return None
    return float(value)


def load_chain_csv(path: str | Path,
                   spot: float,
                   risk_free: float=  0.05,
                   div_yield: float=  0.0,
                   as_of: date | None=  None,
                   clean: bool=  True) -> tuple[VolSurface, list[RejectionRecord]]:
    """Load a CSV option chain and return a VolSurface + rejection log.

    Returns a (VolSurface, list[RejectionRecord]) tuple.  When clean=False,
    every quote is kept and the rejection list is empty.
    """
    by_T: dict[float, list[Quote]]=  {}
    all_rejected: list[RejectionRecord]=  []

    with open(path, newline="", encoding="utf-8") as f:
        reader=  csv.DictReader(f)
        for row in reader:
            missing=  [f for f in _REQUIRED_FIELDS if f not in row]
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            strike=  float(row["strike"])
            T=  _parse_expiry(row["expiry"], as_of)
            otype=  _parse_option_type(row["option_type"])
            price=  float(row["price"])
            bid=  _safe_float(row.get("bid"))
            ask=  _safe_float(row.get("ask"))

            q=  Quote(strike=strike, option_type=otype, price=price, bid=bid, ask=ask)
            by_T.setdefault(T, []).append(q)

    slices: list[ExpirySlice]=  []
    for T, quotes in by_T.items():
        sl=  ExpirySlice(expiry_time=T, quotes=quotes)
        if clean:
            kept, rejected=  clean_quotes(sl, spot)
            all_rejected.extend(rejected)
            if not kept:
                continue  # drop empty slices after cleaning
            sl=  ExpirySlice(expiry_time=T, quotes=kept)
        slices.append(sl)

    if not slices:
        raise ValueError("No slices survived cleaning")

    return (
        VolSurface(spot=spot, risk_free=risk_free, div_yield=div_yield, slices=slices),
        all_rejected,
    )
