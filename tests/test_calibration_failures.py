from arbfree_vol.calibration import calibrate_surface
from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface


def test_too_few_iv_points_is_an_explicit_failed_expiry() -> None:
    quotes = [
        Quote(strike=strike, option_type=OptionType.CALL, price=price)
        for strike, price in ((90, 12), (100, 7), (110, 3), (120, 1))
    ]
    surface = VolSurface(
        spot=100,
        risk_free=0.02,
        div_yield=0,
        slices=[ExpirySlice(expiry_time=0.5, quotes=quotes)],
    )
    report = calibrate_surface(surface)
    assert report.failed_slices == (0.5,)
    assert report.raw_svi_failed_slices == (0.5,)
    assert not report.certificate.certified
