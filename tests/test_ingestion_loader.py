"""Tests for the CSV option chain loader."""
import tempfile
from datetime import date
from pathlib import Path

from arbfree_vol.ingestion.loader import load_chain_csv


SPOT = 100.0
T = 0.5
AS_OF = date(2024, 1, 15)


def _write_csv(content: str) -> Path:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.close()
    return Path(f.name)


def test_load_minimal_chain() -> None:
    csv = """strike,expiry,option_type,price
100.0,2024-07-15,call,10.0
110.0,2024-07-15,call,5.0
100.0,2024-07-15,put,3.0
"""
    path = _write_csv(csv)
    surface, rejected = load_chain_csv(
        path, spot=SPOT, as_of=AS_OF, clean=False
    )

    assert len(surface.slices) == 1
    assert len(surface.slices[0].quotes) == 3
    assert len(rejected) == 0


def test_load_chain_with_bid_ask() -> None:
    csv = """strike,expiry,option_type,price,bid,ask
100.0,2024-07-15,call,10.0,9.5,10.5
110.0,2024-07-15,call,5.0,4.8,5.2
"""
    path = _write_csv(csv)
    surface, rejected = load_chain_csv(
        path, spot=SPOT, as_of=AS_OF, clean=False
    )

    assert surface.slices[0].quotes[0].bid == 9.5
    assert surface.slices[0].quotes[0].ask == 10.5


def test_load_chain_groups_by_expiry() -> None:
    csv = """strike,expiry,option_type,price
100.0,2024-07-15,call,10.0
100.0,2024-10-15,call,12.0
"""
    path = _write_csv(csv)
    surface, _ = load_chain_csv(
        path, spot=SPOT, as_of=AS_OF, clean=False
    )

    assert len(surface.slices) == 2
    Ts = sorted(s.expiry_time for s in surface.slices)
    assert Ts[0] < Ts[1]


def test_load_chain_with_cleaning_rejects_bad() -> None:
    csv = """strike,expiry,option_type,price,bid,ask
100.0,2024-07-15,call,10.0,9.0,11.0
110.0,2024-07-15,call,5.0,6.0,5.0
120.0,2024-07-15,call,-1.0,,
"""
    path = _write_csv(csv)
    surface, rejected = load_chain_csv(
        path, spot=SPOT, as_of=AS_OF, clean=True
    )

    assert len(surface.slices) == 1
    assert len(surface.slices[0].quotes) == 1
    assert surface.slices[0].quotes[0].strike == 100.0
    assert len(rejected) == 2


def test_load_chain_preserves_surface_metadata() -> None:
    csv = """strike,expiry,option_type,price
100.0,2024-07-15,call,10.0
"""
    path = _write_csv(csv)
    surface, _ = load_chain_csv(
        path, spot=SPOT, risk_free=0.04, div_yield=0.02,
        as_of=AS_OF, clean=False
    )

    assert surface.spot == SPOT
    assert surface.risk_free == 0.04
    assert surface.div_yield == 0.02
