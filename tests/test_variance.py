"""Tests for variance.py — per-slice total variance extraction."""
from datetime import date

from pytest import approx

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType, ImpliedVolInput, OptionContract
from arbfree_vol.pricing.black_scholes import price_floats
from arbfree_vol.pricing.implied_vol import implied_vol
from arbfree_vol.variance import slice_total_variance


def test_slice_total_variance_averages_call_put_same_strike() -> None:
    """When call and put at the same strike have *different* implied vols,
    the returned per-strike w must be the average of the two — not equal
    to either one alone."""
    spot = 100.0
    T = 0.5
    r = 0.05
    q = 0.0

    strike = 110.0
    call_sigma = 0.25
    put_sigma = 0.30  # deliberately different

    call_price = price_floats(spot, strike, T, r, q, call_sigma, is_call=True)
    put_price = price_floats(spot, strike, T, r, q, put_sigma, is_call=False)

    call_q = Quote(strike=strike, option_type=OptionType.CALL, price=call_price)
    put_q = Quote(strike=strike, option_type=OptionType.PUT, price=put_price)

    slice_ = ExpirySlice(expiry_time=T, quotes=[call_q, put_q])
    surface = VolSurface(spot=spot, risk_free=r, div_yield=q, slices=[slice_])

    result = slice_total_variance(surface, slice_)

    # Compute expected average from the *actual* recovered implied vols
    call_iv = implied_vol(ImpliedVolInput(
        contract=OptionContract(symbol="_", option_type=OptionType.CALL,
                                strike=strike, expiry_date=date(2004, 1, 1)),
        spot=spot, expiry_time=T, risk_free=r, div_yield=q,
        market_price=call_price))
    put_iv = implied_vol(ImpliedVolInput(
        contract=OptionContract(symbol="_", option_type=OptionType.PUT,
                                strike=strike, expiry_date=date(2004, 1, 1)),
        spot=spot, expiry_time=T, risk_free=r, div_yield=q,
        market_price=put_price))

    w_call = call_iv ** 2 * T
    w_put = put_iv ** 2 * T
    expected = (w_call + w_put) / 2.0

    assert strike in result, "Strike missing from output"
    assert result[strike] == approx(expected, rel=1e-12), (
        f"Expected avg={expected:.15f}, got {result[strike]:.15f}"
    )
    assert result[strike] != approx(w_call, abs=1e-12), (
        "Should not equal call-only value"
    )
    assert result[strike] != approx(w_put, abs=1e-12), (
        "Should not equal put-only value"
    )


def test_slice_total_variance_single_option() -> None:
    """Single call at a strike should give its own w."""
    spot = 100.0
    T = 0.5
    r = 0.05
    q = 0.0

    strike = 100.0
    sigma = 0.20
    price = price_floats(spot, strike, T, r, q, sigma, is_call=True)
    call_q = Quote(strike=strike, option_type=OptionType.CALL, price=price)

    slice_ = ExpirySlice(expiry_time=T, quotes=[call_q])
    surface = VolSurface(spot=spot, risk_free=r, div_yield=q, slices=[slice_])

    result = slice_total_variance(surface, slice_)
    expected = sigma ** 2 * T
    assert result[strike] == approx(expected, abs=1e-8)
