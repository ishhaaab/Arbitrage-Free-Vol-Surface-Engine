"""Tests for the forward curve estimator."""
import logging
from datetime import date
from math import exp, log
from pytest import approx

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.forward import estimate_forward_curve, populate_per_slice_r


SPOT = 100.0
R = 0.05
T = 1.0
_DUMMY_DATE = date(2030, 1, 1)


def _call_price(strike: float, sigma: float = 0.2, tt: float = T) -> float:
    from arbfree_vol.models.option import OptionContract, BlackScholesInput
    from arbfree_vol.pricing.black_scholes import price

    contract = OptionContract(
        symbol="X", option_type=OptionType.CALL, strike=strike,
        expiry_date=_DUMMY_DATE,
    )
    model = BlackScholesInput(
        contract=contract, spot=SPOT, expiry_time=tt,
        risk_free=R, div_yield=0.0, volatility=sigma,
    )
    return price(model)


def _put_price(strike: float, sigma: float = 0.2, tt: float = T) -> float:
    from arbfree_vol.models.option import OptionContract, BlackScholesInput
    from arbfree_vol.pricing.black_scholes import price

    contract = OptionContract(
        symbol="X", option_type=OptionType.PUT, strike=strike,
        expiry_date=_DUMMY_DATE,
    )
    model = BlackScholesInput(
        contract=contract, spot=SPOT, expiry_time=tt,
        risk_free=R, div_yield=0.0, volatility=sigma,
    )
    return price(model)


def test_fwd_curve_recovers_theoretical_forward() -> None:
    # With q=0, the forward should be S * exp(r * T) = 100 * exp(0.05).
    # If both C and P are priced at the same vol (consistent), parity
    # should recover this forward.
    slice_ = ExpirySlice(
        expiry_time=T,
        quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=_call_price(100.0)),
            Quote(strike=100.0, option_type=OptionType.PUT, price=_put_price(100.0)),
            Quote(strike=110.0, option_type=OptionType.CALL, price=_call_price(110.0)),
            Quote(strike=110.0, option_type=OptionType.PUT, price=_put_price(110.0)),
        ],
    )
    surface = VolSurface(spot=SPOT, risk_free=R, div_yield=0.0, slices=[slice_])

    curve = estimate_forward_curve(surface)

    assert T in curve
    assert curve[T] == approx(SPOT * exp(R * T), abs=0.02)  # within 2%


def test_fwd_curve_fallback_when_no_call_put_pairs() -> None:
    # Only calls — no way to extract F from parity, so we fallback to q=0.
    slice_ = ExpirySlice(
        expiry_time=T,
        quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=10.0),
            Quote(strike=110.0, option_type=OptionType.CALL, price=5.0),
        ],
    )
    surface = VolSurface(spot=SPOT, risk_free=R, div_yield=0.0, slices=[slice_])

    curve = estimate_forward_curve(surface)

    assert curve[T] == SPOT * exp(R * T)


def test_fwd_curve_multiple_slices() -> None:
    slices = [
        ExpirySlice(
            expiry_time=0.5,
            quotes=[
                Quote(strike=100.0, option_type=OptionType.CALL, price=_call_price(100.0, sigma=0.2, tt=0.5)),
                Quote(strike=100.0, option_type=OptionType.PUT, price=_put_price(100.0, sigma=0.2, tt=0.5)),
            ],
        ),
        ExpirySlice(
            expiry_time=1.0,
            quotes=[
                Quote(strike=100.0, option_type=OptionType.CALL, price=_call_price(100.0, sigma=0.2)),
                Quote(strike=100.0, option_type=OptionType.PUT, price=_put_price(100.0, sigma=0.2)),
            ],
        ),
    ]
    surface = VolSurface(spot=SPOT, risk_free=R, div_yield=0.0, slices=slices)

    curve = estimate_forward_curve(surface)

    assert sorted(curve.keys()) == [0.5, 1.0]
    for T, F in curve.items():
        assert F == approx(SPOT * exp(R * T), abs=0.02)


def test_populate_per_slice_r_uses_per_slice_q() -> None:
    """Verify populate_per_slice_r uses per-slice div_yield, not surface-level."""
    # Surface with q=0.05, but slice has q=0.02 (per-slice override)
    slice_ = ExpirySlice(
        expiry_time=1.0,
        div_yield=0.02,  # per-slice override
        quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=10.45),
            Quote(strike=100.0, option_type=OptionType.PUT, price=5.57),
        ],
    )
    surface = VolSurface(spot=100.0, risk_free=0.05, div_yield=0.05, slices=[slice_])

    fwd_curve = {1.0: 105.0}  # F = 105
    populate_per_slice_r(surface, fwd_curve)

    # r = log(F/S)/T + q_slice = log(105/100)/1 + 0.02
    # = 0.04879 + 0.02 = 0.06879
    # NOT log(F/S)/T + q_surface = 0.04879 + 0.05 = 0.09879
    expected_r = log(105.0 / 100.0) / 1.0 + 0.02
    assert slice_.risk_free == approx(expected_r, abs=1e-6)


def test_fwd_curve_fallback_logs_default_substitution(caplog) -> None:
    """Fallback with the r=0.05/q=0.0 substitution defaults must say so.

    The provenance matters: when a slice has no (call, put) parity pair,
    the theoretical forward uses whatever r/q the surface carries.  If
    those are the ingestion-layer substitution defaults, a reader must
    not mistake them for genuinely observed rates.

    The wording is a documented HEURISTIC (see
    ``estimate_forward_curve``): r/q matching the default constants is
    INFERRED to indicate substitution, but the same values can be
    genuinely observed — the log says "may be an observed value" and
    never asserts provenance.
    """
    slice_ = ExpirySlice(
        expiry_time=T,
        quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=10.0),
            Quote(strike=110.0, option_type=OptionType.CALL, price=5.0),
        ],
    )
    surface = VolSurface(spot=SPOT, risk_free=0.05, div_yield=0.0, slices=[slice_])

    with caplog.at_level(logging.WARNING, logger="arbfree_vol.forward"):
        estimate_forward_curve(surface)

    assert "default substitution" in caplog.text
    # The heuristic wording: the values MATCH the defaults but may be
    # observed — provenance is inferred, not recorded.
    assert "may be an observed value" in caplog.text
    assert "provenance is inferred by this heuristic" in caplog.text


def test_fwd_curve_fallback_logs_non_default_values(caplog) -> None:
    """Fallback with a genuine (non-default) q must say r/q are non-default."""
    slice_ = ExpirySlice(
        expiry_time=T,
        quotes=[
            Quote(strike=100.0, option_type=OptionType.CALL, price=10.0),
            Quote(strike=110.0, option_type=OptionType.CALL, price=5.0),
        ],
    )
    surface = VolSurface(spot=SPOT, risk_free=0.05, div_yield=0.02, slices=[slice_])

    with caplog.at_level(logging.WARNING, logger="arbfree_vol.forward"):
        estimate_forward_curve(surface)

    assert "non-default" in caplog.text
