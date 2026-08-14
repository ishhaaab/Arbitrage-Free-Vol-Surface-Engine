"""Tests for the repair report dataclasses.

``arbfree_vol.repair.report`` is the pure-data contract layer of the repair
pipeline: it carries the rejected quotes, per-slice fit outcomes, remaining
arbitrage violations, and aggregate metrics between the engine and the
iteration loop / downstream consumers.  Because it is a shared contract with
several dependents, the dataclass behaviour itself is pinned here:

- ``RejectedQuote``: field round-trip, frozen immutability, slot layout.
- ``RepairMetrics``: ``rejection_rate`` normal / full / zero-rejected paths,
  plus the zero-total else-branch (``n_total_quotes == 0`` -> exactly 0.0,
  never a ``ZeroDivisionError``).
- ``RepairReport``: full explicit construction, default values, mutable-list
  ``default_factory`` independence, frozen immutability, slot layout.

All tests are pure dataclass construction — no network, no calibration.
"""
from dataclasses import FrozenInstanceError

import pytest

from arbfree_vol.models.fitted import (
    FittedSlice,
    FittedSSVISlice,
    FittedSABRSlice,
)
from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.arbitrage.report import (
    ArbitrageReport,
    ArbitrageViolation,
    ViolationType,
)
from arbfree_vol.repair.report import RejectedQuote, RepairMetrics, RepairReport
from arbfree_vol.sabr.model import SABRParams
from arbfree_vol.ssvi.model import SSVIParams, eSSVISurfaceParams
from arbfree_vol.svi.model import SVIParams


def _rejected_quote() -> RejectedQuote:
    """A canned rejected quote used across the frozen/slots checks."""
    return RejectedQuote(
        strike=110.0,
        expiry_time=1.0,
        option_type=OptionType.CALL,
        reason=ViolationType.MONOTONICITY,
        magnitude=0.05,
    )


def _metrics() -> RepairMetrics:
    """Metrics with a non-trivial rejection rate (3 of 10 rejected)."""
    return RepairMetrics(
        n_rejected=3,
        n_total_quotes=10,
        n_slices_input=2,
        n_slices_fitted=2,
        n_violations_before=4,
        n_violations_after=0,
    )


def _zero_metrics() -> RepairMetrics:
    """All-zero metrics used for minimal reports."""
    return RepairMetrics(
        n_rejected=0, n_total_quotes=0,
        n_slices_input=0, n_slices_fitted=0,
        n_violations_before=0, n_violations_after=0,
    )


def _surface() -> VolSurface:
    """A minimal single-slice surface for ``cleaned_surface``."""
    return VolSurface(
        spot=100.0,
        risk_free=0.05,
        div_yield=0.0,
        slices=[ExpirySlice(
            expiry_time=1.0,
            quotes=[Quote(
                strike=100.0,
                option_type=OptionType.CALL,
                price=10.0,
            )],
        )],
    )


def _fitted_slices() -> tuple[FittedSlice, ...]:
    """One raw-SVI fitted slice."""
    return (
        FittedSlice(
            expiry_time=1.0,
            params=SVIParams(a=0.02, b=0.3, rho=-0.3, m=0.0, sigma=0.2),
            rmse=0.01,
            forward_price=100.0,
            n_quotes_total=5,
            n_quotes_used=5,
        ),
    )


def _fitted_ssvi_slices() -> tuple[FittedSSVISlice, ...]:
    """One eSSVI-fitted slice (with surface-level eSSVI params)."""
    return (
        FittedSSVISlice(
            expiry_time=1.0,
            ssvi=SSVIParams(theta=0.04, rho=-0.3, psi=0.4),
            rmse=0.01,
            forward_price=100.0,
            n_quotes_total=5,
            n_quotes_used=5,
            essvi=eSSVISurfaceParams(eta=0.4, gamma=0.5),
        ),
    )


def _fitted_sabr_slices() -> tuple[FittedSABRSlice, ...]:
    """One SABR-fitted slice."""
    return (
        FittedSABRSlice(
            expiry_time=1.0,
            sabr=SABRParams(alpha=0.2, beta=0.5, rho=-0.3, nu=0.4),
            rmse=0.01,
            forward_price=100.0,
            n_quotes_total=5,
            n_quotes_used=5,
        ),
    )


def _minimal_report() -> RepairReport:
    """A report with only the required fields (no defaults exercised)."""
    return RepairReport(
        rejected=(),
        fitted_slices=(),
        remaining_violations=ArbitrageReport(violations=[]),
        metrics=_zero_metrics(),
        cleaned_surface=None,
    )


def _full_report() -> RepairReport:
    """A report exercising every explicit field, including the defaults
    overridden to non-default values."""
    return RepairReport(
        rejected=(_rejected_quote(),),
        fitted_slices=_fitted_slices(),
        remaining_violations=ArbitrageReport(violations=[
            ArbitrageViolation(
                kind=ViolationType.CALENDAR,
                detail="fake remaining violation",
                magnitude=0.1,
            ),
        ]),
        metrics=_metrics(),
        cleaned_surface=_surface(),
        fitted_ssvi_slices=_fitted_ssvi_slices(),
        fitted_sabr_slices=_fitted_sabr_slices(),
        repair_infeasible=True,
        fallback_slices=[100.0, 110.0],
        failed_slices=[120.0],
        sabr_mapping_failed_slices=[130.0],
    )


# ── RejectedQuote tests ───────────────────────────────────────────────────────


def test_rejected_quote_round_trips_all_fields() -> None:
    quote = _rejected_quote()

    assert quote == RejectedQuote(
        strike=110.0,
        expiry_time=1.0,
        option_type=OptionType.CALL,
        reason=ViolationType.MONOTONICITY,
        magnitude=0.05,
    )
    assert quote.strike == 110.0
    assert quote.expiry_time == 1.0
    assert quote.option_type is OptionType.CALL
    assert quote.reason is ViolationType.MONOTONICITY
    assert quote.magnitude == pytest.approx(0.05)


def test_rejected_quote_is_frozen() -> None:
    quote = _rejected_quote()

    with pytest.raises(FrozenInstanceError):
        quote.strike = 120.0


def test_rejected_quote_has_slots() -> None:
    assert not hasattr(_rejected_quote(), "__dict__")


# ── RepairMetrics tests ───────────────────────────────────────────────────────


def test_rejection_rate_normal_path() -> None:
    metrics = _metrics()

    assert metrics.rejection_rate == pytest.approx(0.3)


def test_rejection_rate_full_rejection() -> None:
    metrics = RepairMetrics(
        n_rejected=10, n_total_quotes=10,
        n_slices_input=0, n_slices_fitted=0,
        n_violations_before=0, n_violations_after=0,
    )

    assert metrics.rejection_rate == pytest.approx(1.0)


def test_rejection_rate_zero_rejected() -> None:
    metrics = RepairMetrics(
        n_rejected=0, n_total_quotes=10,
        n_slices_input=0, n_slices_fitted=0,
        n_violations_before=0, n_violations_after=0,
    )

    assert metrics.rejection_rate == pytest.approx(0.0)


def test_rejection_rate_zero_total_does_not_raise() -> None:
    """Pins the else-branch of ``rejection_rate``: an empty quote universe
    yields exactly 0.0 instead of raising ``ZeroDivisionError``."""
    metrics = _zero_metrics()

    assert metrics.rejection_rate == 0.0


def test_repair_metrics_is_frozen() -> None:
    metrics = _metrics()

    with pytest.raises(FrozenInstanceError):
        metrics.n_rejected = 5


def test_repair_metrics_has_slots() -> None:
    assert not hasattr(_metrics(), "__dict__")


# ── RepairReport tests ────────────────────────────────────────────────────────


def test_repair_report_round_trips_all_explicit_fields() -> None:
    report = _full_report()

    assert report.rejected == (_rejected_quote(),)
    assert report.fitted_slices == _fitted_slices()
    assert report.remaining_violations == ArbitrageReport(violations=[
        ArbitrageViolation(
            kind=ViolationType.CALENDAR,
            detail="fake remaining violation",
            magnitude=0.1,
        ),
    ])
    assert report.remaining_violations.is_arbitrage_free is False
    assert report.metrics == _metrics()
    assert report.cleaned_surface == _surface()
    assert report.fitted_ssvi_slices == _fitted_ssvi_slices()
    assert report.fitted_sabr_slices == _fitted_sabr_slices()
    assert report.repair_infeasible is True
    assert report.fallback_slices == [100.0, 110.0]
    assert report.failed_slices == [120.0]
    assert report.sabr_mapping_failed_slices == [130.0]


def test_repair_report_defaults() -> None:
    report = _minimal_report()

    assert report.fitted_ssvi_slices == ()
    assert report.fitted_sabr_slices == ()
    assert report.repair_infeasible is False
    assert report.fallback_slices == []
    assert report.failed_slices == []
    assert report.sabr_mapping_failed_slices == []


def test_repair_report_default_factory_lists_are_independent() -> None:
    first = _minimal_report()
    second = _minimal_report()

    first.fallback_slices.append(100.0)
    first.failed_slices.append(110.0)
    first.sabr_mapping_failed_slices.append(120.0)

    assert second.fallback_slices == []
    assert second.failed_slices == []
    assert second.sabr_mapping_failed_slices == []


def test_repair_report_is_frozen() -> None:
    report = _minimal_report()

    with pytest.raises(FrozenInstanceError):
        report.rejected = ()


def test_repair_report_has_slots() -> None:
    assert not hasattr(_minimal_report(), "__dict__")
