"""Load a dated European option-chain CSV using ACT/365F."""

import csv
from datetime import date, datetime
from pathlib import Path

from arbfree_vol.ingestion.cleaning import RejectionRecord, clean_quotes
from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface

_REQUIRED_FIELDS = ("strike", "expiry", "option_type", "price")


def _parse_option_type(value: str) -> OptionType:
    normalized = value.strip().lower()
    if normalized in ("call", "c"):
        return OptionType.CALL
    if normalized in ("put", "p"):
        return OptionType.PUT
    raise ValueError(f"Unknown option type: {value!r}")


def _optional_float(value: str | None) -> float | None:
    return None if value in (None, "") else float(value)


def load_chain_csv(
    path: str | Path,
    *,
    spot: float,
    as_of: date,
    risk_free: float,
    div_yield: float = 0.0,
    clean: bool = True,
) -> tuple[VolSurface, list[RejectionRecord]]:
    """Return a surface and cleaning audit from the documented CSV schema."""
    by_expiry: dict[float, list[Quote]] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in _REQUIRED_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        for row in reader:
            expiry = datetime.strptime(row["expiry"], "%Y-%m-%d").date()
            days = (expiry - as_of).days
            if days <= 0:
                raise ValueError(f"Option expiry {expiry.isoformat()} is not after as_of")
            maturity = days / 365.0
            quote = Quote(
                strike=float(row["strike"]),
                option_type=_parse_option_type(row["option_type"]),
                price=float(row["price"]),
                bid=_optional_float(row.get("bid")),
                ask=_optional_float(row.get("ask")),
            )
            by_expiry.setdefault(maturity, []).append(quote)

    slices: list[ExpirySlice] = []
    rejected: list[RejectionRecord] = []
    for maturity, quotes in sorted(by_expiry.items()):
        expiry_slice = ExpirySlice(expiry_time=maturity, quotes=quotes)
        if clean:
            result = clean_quotes(
                expiry_slice,
                spot,
                risk_free=risk_free,
                div_yield=div_yield,
            )
            rejected.extend(result.rejected_quotes)
            if not result.accepted_quotes:
                continue
            expiry_slice = ExpirySlice(
                expiry_time=maturity, quotes=list(result.accepted_quotes)
            )
        slices.append(expiry_slice)

    if not slices:
        raise ValueError("No slices survived cleaning")
    return VolSurface(
        spot=spot,
        risk_free=risk_free,
        div_yield=div_yield,
        slices=slices,
    ), rejected
