from dataclasses import replace

from arbfree_vol.models.fitted import FittedSlice
from arbfree_vol.svi.model import SVIParams
from arbfree_vol.verification import verify_surface


def _slice(maturity: float, variance: float) -> FittedSlice:
    return FittedSlice(
        expiry_time=maturity,
        params=SVIParams(a=variance, b=0, rho=0, m=0, sigma=0.2),
        rmse=0,
        forward_price=100,
        n_quotes_total=5,
        n_quotes_used=5,
    )


def test_certifies_safe_surface_on_declared_grid() -> None:
    result = verify_surface([_slice(0.5, 0.02), _slice(1, 0.04)])
    assert result.certified
    assert result.min_total_variance == 0.02
    assert result.min_butterfly_density > 0
    assert result.min_calendar_margin == 0.02
    assert (result.k_min, result.k_max, result.grid_size) == (-1.5, 1.5, 241)


def test_rejects_negative_variance() -> None:
    result = verify_surface([_slice(0.5, -0.01)])
    assert not result.certified
    assert result.min_total_variance < 0


def test_rejects_calendar_crossing() -> None:
    result = verify_surface([_slice(0.5, 0.04), _slice(1, 0.02)])
    assert not result.certified
    assert result.min_calendar_margin == -0.02


def test_fallback_or_failure_prevents_certification() -> None:
    safe = verify_surface([_slice(0.5, 0.02)])
    assert not replace(safe, certified=False).certified
    assert not verify_surface([_slice(0.5, 0.02)], has_fallbacks=True).certified
    assert not verify_surface([_slice(0.5, 0.02)], has_failures=True).certified


def test_empty_fit_is_not_certified() -> None:
    assert not verify_surface([]).certified
