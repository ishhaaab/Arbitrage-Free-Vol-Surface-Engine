"""Tests for Black-Scholes Greeks."""

from datetime import date

from pytest import approx

from arbfree_vol.models.option import BlackScholesInput, OptionContract, OptionType
from arbfree_vol.pricing.greeks import greeks


def _model(option_type: OptionType) -> BlackScholesInput:
    contract = OptionContract(
        symbol="NVDA",
        option_type=option_type,
        strike=100,
        expiry=date(2026, 11, 27),
    )
    return BlackScholesInput(
        contract=contract,
        spot=100,
        expiry_time=1,
        risk_free=0.05,
        div_yield=0,
        volatility=0.2,
    )


def test_call_greeks_match_known_values() -> None:
    g = greeks(_model(OptionType.CALL))

    assert g.delta == approx(0.636831, abs=1e-6)
    assert g.gamma == approx(0.018762, abs=1e-6)
    assert g.vega == approx(37.524035, abs=1e-6)
    assert g.theta == approx(-6.414028, abs=1e-6)
    assert g.rho == approx(53.232482, abs=1e-6)


def test_put_greeks_match_known_values() -> None:
    g = greeks(_model(OptionType.PUT))

    assert g.delta == approx(-0.363169, abs=1e-6)
    assert g.gamma == approx(0.018762, abs=1e-6)
    assert g.vega == approx(37.524035, abs=1e-6)
    assert g.theta == approx(-1.657880, abs=1e-6)
    assert g.rho == approx(-41.890461, abs=1e-6)


def test_gamma_and_vega_are_identical_for_call_and_put() -> None:
    call = greeks(_model(OptionType.CALL))
    put = greeks(_model(OptionType.PUT))

    assert call.gamma == approx(put.gamma)
    assert call.vega == approx(put.vega)


def test_put_call_delta_parity() -> None:
    # delta_call - delta_put == exp(-q*T); q = 0 here, so == 1.0
    call = greeks(_model(OptionType.CALL))
    put = greeks(_model(OptionType.PUT))

    assert call.delta - put.delta == approx(1.0, abs=1e-9)
