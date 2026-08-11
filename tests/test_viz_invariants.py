"""Targeted invariant tests for the fallback-mask in viz plotting.

The smoke tests in ``test_viz.py`` only assert ``fig.axes is not None``.
These two tests pin down the actual mask data structure produced by
``plot_iv_heatmap`` (``viz/surface.py``): when eSSVI falls back for a
maturity slice, the whole maturity row of the heatmap is grayed out by
being turned into NaN and then masked via ``np.ma.masked_invalid``.

The fallback maturity is placed exactly on the heatmap's maturity grid
(``np.linspace(0.5, 1.0, 5)`` == ``[0.5, 0.625, 0.75, 0.875, 1.0]``),
so the expected masked row is known by construction.
"""

from datetime import date

import matplotlib
matplotlib.use("Agg")

import numpy as np

from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
from arbfree_vol.models.option import OptionType
from arbfree_vol.repair.engine import repair
from arbfree_vol.surface.interpolate import build_fitted_surface
from arbfree_vol.viz.surface import plot_iv_heatmap

_DUMMY = date(2030, 1, 1)
SPOT = 100.0
R = 0.05

# Fallback maturity that sits exactly on a heatmap grid row.
FALLBACK_T = 0.75
N_MATURITIES = 5
N_STRIKES = 5
_TOL = 0.01  # make_fallback_mask tolerance


def _bp(otype, K, sigma=0.2, tt=1.0):
    from arbfree_vol.models.option import OptionContract, BlackScholesInput
    from arbfree_vol.pricing.black_scholes import price
    c = OptionContract(symbol="X", option_type=otype, strike=K, expiry_date=_DUMMY)
    m = BlackScholesInput(contract=c, spot=SPOT, expiry_time=tt,
                          risk_free=R, div_yield=0.0, volatility=sigma)
    return price(m)


def _fitted_surface():
    strikes = [80, 90, 100, 110, 120]
    qs1 = [Quote(strike=K, option_type=o, price=_bp(o, K, tt=0.5))
           for K in strikes for o in [OptionType.CALL, OptionType.PUT]]
    qs2 = [Quote(strike=K, option_type=o, price=_bp(o, K, tt=1.0))
           for K in strikes for o in [OptionType.CALL, OptionType.PUT]]
    s = VolSurface(spot=SPOT, risk_free=R, div_yield=0.0, slices=[
        ExpirySlice(expiry_time=0.5, quotes=qs1),
        ExpirySlice(expiry_time=1.0, quotes=qs2),
    ])
    r = repair(s)
    return build_fitted_surface(r)


def _heatmap_mesh_array(fallback_slices):
    """Return the masked array feeding ``plot_iv_heatmap``'s QuadMesh."""
    fs = _fitted_surface()
    fig = plot_iv_heatmap(
        fs,
        n_strikes=N_STRIKES,
        n_maturities=N_MATURITIES,
        fallback_slices=fallback_slices,
    )
    mesh = fig.axes[0].collections[0]
    return mesh.get_array()


def test_fallback_mask_greys_exactly_the_fallback_maturity_row() -> None:
    """The mask marks only the fallback maturity's row — nothing else.

    ``plot_iv_heatmap`` computes ``make_fallback_mask(maturities,
    fallback_slices)``, broadcasts it across strikes, NaNs those cells
    and calls ``np.ma.masked_invalid``.  The ``QuadMesh`` array's
    boolean ``.mask`` is that exact 2-D structure, so it must equal the
    expected ``(fallback_row, all_strikes)`` block and be ``False``
    everywhere else.
    """
    arr = _heatmap_mesh_array([FALLBACK_T])

    maturities = np.linspace(0.5, 1.0, N_MATURITIES)
    expected_row_1d = np.abs(maturities - FALLBACK_T) <= _TOL
    expected_2d = np.broadcast_to(
        expected_row_1d[:, None], (N_MATURITIES, N_STRIKES)
    )

    assert arr.shape == (N_MATURITIES, N_STRIKES)
    assert arr.mask.shape == (N_MATURITIES, N_STRIKES)
    # The fallback T is on the grid, so exactly one row is masked.
    assert expected_row_1d.sum() == 1
    assert np.array_equal(arr.mask, expected_2d)


def test_no_nan_or_inf_outside_the_fallback_mask() -> None:
    """Every plotted value outside the fallback mask is finite.

    Cells outside the masked region are the ones actually rendered with
    the plasma colormap; if any of them were NaN/Inf the heatmap would
    silently drop or mis-colour data.  The fallback mask (the row of
    ``True`` values) must be the *only* place non-finite values appear.
    """
    arr = _heatmap_mesh_array([FALLBACK_T])

    mask = arr.mask
    # The mask must actually mark the fallback region (guard against a
    # vacuous pass if the plotting code stopped masking anything).
    assert mask.any()
    # Every value outside the fallback mask is a rendered, finite cell.
    assert np.all(np.isfinite(arr[~mask]))
