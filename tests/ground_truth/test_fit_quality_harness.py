"""Fit-quality harness demonstration on a synthetic chain.

Builds a one-slice quote chain from a known SVI smile, adds a small
bid/ask noise, computes mid implied vols with the harness, fits the
correctly-specified model (SVI) and a wrong model (flat vol), and shows the
harness's per-slice IV RMSE can DISTINGUISH the two: the correctly-specified
model's RMSE is small (documented threshold), the wrong model's is
materially larger.  This is a fit-quality gate for synthetic chains, not a
correctness gate for the arbitrage machinery (see ``fit_quality.py``).
"""

from __future__ import annotations

import math

import pytest
from pytest import approx

from arbfree_vol.svi.calibration import calibrate
from arbfree_vol.svi.model import svi_total_variance

from tests.ground_truth.arbitrage_cases import SVI_ARB_FREE_FROM_ESSVI
from tests.ground_truth.cases import build_svi_quote_surface
from tests.ground_truth.fit_quality import (
    per_strike_iv_errors,
    quote_mid_price,
    slice_iv_rmse,
    slice_mid_ivs,
)

# Documented thresholds (implied-vol units).
_SVI_RMSE_MAX: float = 0.005
"""Correctly-specified model on a clean synthetic chain: the SVI fit of the
mid total variances should reproduce the smile to well under 0.005 IV (the
achieved value is ~1e-4; the threshold is 50x headroom)."""

_FLAT_RMSE_MIN: float = 0.02
"""Wrong model (flat vol) on the same chain: the smile spans ~0.61..0.69 IV,
so a flat fit at the ATM IV has RMSE ~0.025-0.03 — materially larger."""

_HALF_SPREAD: float = 0.005
"""Half of the relative bid-ask spread of the synthetic chain."""


def _model_iv_closure(surface, sl, fit):
    """Model-IV closure for a fitted SVI slice (absolute-strike -> IV)."""
    from math import log, sqrt

    from arbfree_vol.models.surface import get_q, get_r

    F = surface.spot * math.exp(
        (get_r(surface, sl) - get_q(surface, sl)) * sl.expiry_time
    )
    T = sl.expiry_time

    def model_iv(K: float) -> float:
        k = log(K / F)
        return sqrt(svi_total_variance(k, fit.a, fit.b, fit.rho, fit.m,
                                       fit.sigma) / T)

    return model_iv


def test_harness_distinguishes_correct_from_wrong_model() -> None:
    """Correct SVI fit has small per-slice RMSE; flat vol has a large one."""
    surface = build_svi_quote_surface(
        1.0, SVI_ARB_FREE_FROM_ESSVI.params,
        n_k=15, k_lo=-0.5, k_hi=0.5,
    )
    sl = surface.slices[0]
    T = sl.expiry_time
    F = surface.spot * math.exp((surface.risk_free - surface.div_yield) * T)

    # 1. Market mid IVs via the harness.
    mid_ivs = slice_mid_ivs(surface, sl)
    assert len(mid_ivs) == len({q.strike for q in sl.quotes}), (
        "every strike must yield a solvable mid IV on a clean chain"
    )

    # 2. Fit the correctly-specified model to the mid total variances.
    points = [
        (math.log(K / F), iv * iv * T)
        for K, iv in sorted(mid_ivs.items())
    ]
    fit = calibrate(points)
    svi_model_iv = _model_iv_closure(surface, sl, fit)

    # 3. Wrong model: flat vol pinned at the ATM mid IV.
    atm_iv = mid_ivs[F]  # strike == forward is on the grid (k=0)
    flat_model_iv = lambda _K: atm_iv  # noqa: E731

    svi_rmse = slice_iv_rmse(surface, sl, svi_model_iv)
    flat_rmse = slice_iv_rmse(surface, sl, flat_model_iv)

    # Per-strike errors are exposed for inspection.
    errors = per_strike_iv_errors(surface, sl, svi_model_iv)
    assert len(errors) == len(mid_ivs)
    assert all(isinstance(e[3], float) for e in errors)

    assert svi_rmse < _SVI_RMSE_MAX, (
        f"correctly-specified model per-slice IV RMSE {svi_rmse:.5f} must be "
        f"below the documented {_SVI_RMSE_MAX}"
    )
    assert flat_rmse > _FLAT_RMSE_MIN, (
        f"wrong (flat) model per-slice IV RMSE {flat_rmse:.5f} must exceed "
        f"the documented {_FLAT_RMSE_MIN}"
    )
    assert flat_rmse > 4.0 * svi_rmse, (
        f"the wrong model's RMSE ({flat_rmse:.5f}) must be materially larger "
        f"than the correct model's ({svi_rmse:.5f}) — the harness must "
        f"distinguish fit quality"
    )


def test_quote_mid_price_uses_bid_ask() -> None:
    """The mid-price helper prefers bid/ask and falls back to price."""
    surface = build_svi_quote_surface(
        1.0, SVI_ARB_FREE_FROM_ESSVI.params, n_k=9, k_lo=-0.4, k_hi=0.4,
    )
    q = surface.slices[0].quotes[0]
    mid = quote_mid_price(q)
    assert mid == approx((q.bid + q.ask) / 2.0, abs=1e-15)
    assert q.bid < mid < q.ask

    from arbfree_vol.models.option import OptionType
    from arbfree_vol.models.surface import Quote

    bare = Quote(strike=q.strike, option_type=OptionType.CALL, price=7.5)
    assert quote_mid_price(bare) == approx(7.5, abs=1e-15)
