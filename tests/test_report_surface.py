from arbfree_vol.models.fitted import FittedSlice
from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import ExpirySlice, Quote, VolSurface
from arbfree_vol.report import CalibrationReport, NumericalCertificate
from arbfree_vol.surface import build_fitted_surface
from arbfree_vol.svi.model import SVIParams


def _report(certified: bool) -> CalibrationReport:
    fitted = FittedSlice(
        expiry_time=0.5,
        params=SVIParams(a=0.02, b=0, rho=0, m=0, sigma=0.2),
        rmse=0,
        forward_price=101,
        n_quotes_total=5,
        n_quotes_used=5,
    )
    surface = VolSurface(
        spot=100,
        risk_free=0.02,
        div_yield=0,
        slices=[
            ExpirySlice(
                expiry_time=0.5,
                quotes=[Quote(strike=100, option_type=OptionType.CALL, price=6)],
            )
        ],
    )
    return CalibrationReport(
        dataset_source="test",
        dataset_timestamp="2026-01-01",
        dependency_versions=(),
        input_quotes=1,
        accepted_quotes=1,
        rejected=(),
        forwards=((0.5, 101),),
        raw_svi_slices=(fitted,),
        raw_svi_failed_slices=(),
        fitted_slices=(fitted,),
        fitted_ssvi_slices=(),
        fallback_slices=(),
        failed_slices=(),
        certificate=NumericalCertificate(-1.5, 1.5, 241, 1e-4, 0.02, 1, None, certified),
        runtime_seconds=0,
        cleaned_surface=surface,
    )


def test_surface_rejects_uncertified_report_by_default() -> None:
    try:
        build_fitted_surface(_report(False))
    except ValueError as error:
        assert "not certified" in str(error)
    else:
        raise AssertionError("uncertified report was accepted")


def test_surface_allows_explicit_unsafe_inspection() -> None:
    assert build_fitted_surface(_report(False), allow_uncertified=True).fitted_slices


def test_surface_accepts_certified_report() -> None:
    assert build_fitted_surface(_report(True)).fitted_slices
