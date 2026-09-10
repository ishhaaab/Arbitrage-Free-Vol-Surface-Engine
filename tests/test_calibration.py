from datetime import date

from arbfree_vol.calibration import calibrate_surface
from arbfree_vol.models.option import BlackScholesInput, OptionContract, OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface
from arbfree_vol.pricing.black_scholes import price
from arbfree_vol.ssvi.model import ssvi_w


def _surface() -> VolSurface:
    spot = 100.0
    rate = 0.02
    slices: list[ExpirySlice] = []
    for maturity, theta in ((0.5, 0.02), (1.0, 0.04)):
        quotes: list[Quote] = []
        forward = spot * __import__("math").exp(rate * maturity)
        for strike in (80, 90, 100, 110, 120):
            k = __import__("math").log(strike / forward)
            volatility = (ssvi_w(k, theta, -0.3, 1.0) / maturity) ** 0.5
            for kind in (OptionType.CALL, OptionType.PUT):
                value = price(
                    BlackScholesInput(
                        contract=OptionContract(
                            symbol="X",
                            option_type=kind,
                            strike=strike,
                            expiry_date=date(2030, 1, 1),
                        ),
                        spot=spot,
                        expiry_time=maturity,
                        risk_free=rate,
                        div_yield=0,
                        volatility=volatility,
                    )
                )
                quotes.append(Quote(strike=strike, option_type=kind, price=value))
        slices.append(ExpirySlice(expiry_time=maturity, quotes=quotes))
    return VolSurface(spot=spot, risk_free=rate, div_yield=0, slices=slices)


def test_calibration_reports_both_models_and_certificate() -> None:
    report = calibrate_surface(
        _surface(), dataset_source="synthetic", dataset_timestamp="2026-01-01"
    )
    assert len(report.raw_svi_slices) == 2
    assert len(report.fitted_ssvi_slices) == 2
    assert report.raw_svi_rmse is not None
    assert report.constrained_ssvi_rmse is not None
    assert report.certificate.certified
    assert report.dataset_source == "synthetic"
    assert dict(report.dependency_versions)["scipy"]


def test_calibration_uses_parity_forward_for_iv_carry_without_mutating_input() -> None:
    surface = _surface()
    surface.div_yield = 0.10
    report = calibrate_surface(surface)
    assert surface.slices[0].div_yield is None
    assert abs(report.cleaned_surface.slices[0].div_yield or 0) < 1e-10
