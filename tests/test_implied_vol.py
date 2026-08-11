"""Tests for the implied volatility solver."""

from datetime import date

import pytest
from pytest import approx
from pydantic import ValidationError

from arbfree_vol.models.option import (
    BlackScholesInput,
    ImpliedVolInput,
    OptionContract,
    OptionType,
)
from arbfree_vol.pricing.black_scholes import price
from arbfree_vol.pricing.implied_vol import implied_vol


def _contract(option_type: OptionType) -> OptionContract:
    return OptionContract(
        symbol="NVDA",
        option_type=option_type,
        strike=100,
        expiry_date=date(2026, 11, 27),
    )


def _iv_input(option_type: OptionType, market_price: float) -> ImpliedVolInput:
    return ImpliedVolInput(
        contract=_contract(option_type),
        spot=100,
        expiry_time=1,
        risk_free=0.05,
        div_yield=0,
        market_price=market_price,
    )


def test_round_trip_recovers_call_volatility() -> None:
    true_sigma = 0.2
    bs = BlackScholesInput(
        contract=_contract(OptionType.CALL),
        spot=100,
        expiry_time=1,
        risk_free=0.05,
        div_yield=0,
        volatility=true_sigma,
    )
    p = price(bs)

    recovered = implied_vol(_iv_input(OptionType.CALL, p))

    assert recovered == approx(true_sigma, abs=1e-6)


def test_round_trip_recovers_put_volatility() -> None:
    true_sigma = 0.35
    bs = BlackScholesInput(
        contract=_contract(OptionType.PUT),
        spot=100,
        expiry_time=1,
        risk_free=0.05,
        div_yield=0,
        volatility=true_sigma,
    )
    p = price(bs)

    recovered = implied_vol(_iv_input(OptionType.PUT, p))

    assert recovered == approx(true_sigma, abs=1e-6)


def test_price_above_no_arbitrage_bound_returns_none() -> None:
    # A call can never be worth more than the discounted spot; 200 is impossible.
    assert implied_vol(_iv_input(OptionType.CALL, 200.0)) is None


def test_price_below_intrinsic_returns_none() -> None:
    # Deep ITM call: spot 100, strike 50 ie intrinsic ~50. A price of 1.0 is below
    # any achievable model price, so no implied vol exists.
    model = ImpliedVolInput(
        contract=OptionContract(
            symbol="NVDA",
            option_type=OptionType.CALL,
            strike=50,
            expiry_date=date(2026, 11, 27),
        ),
        spot=100,
        expiry_time=1,
        risk_free=0.05,
        div_yield=0,
        market_price=1.0,
    )
    assert implied_vol(model) is None


# ---------------------------------------------------------------------------
# Branch-coverage tests for the Newton fast path / Brent fallback
# ---------------------------------------------------------------------------


class TestImpliedVolBranches:
    """Exercise every branch of the Newton/Brent solver and pin the
    documented contracts from the ``implied_vol`` docstring."""

    def _contract(self, option_type: OptionType, strike: float = 100.0) -> OptionContract:
        return OptionContract(
            symbol="NVDA",
            option_type=option_type,
            strike=strike,
            expiry_date=date(2026, 11, 27),
        )

    def _input(
        self, option_type: OptionType, market_price: float,
        strike: float = 100.0, T: float = 1.0,
    ) -> ImpliedVolInput:
        return ImpliedVolInput(
            contract=self._contract(option_type, strike),
            spot=100,
            expiry_time=T,
            risk_free=0.05,
            div_yield=0,
            market_price=market_price,
        )

    def _target_price(self, option_type: OptionType, strike: float, sigma: float) -> float:
        bs = BlackScholesInput(
            contract=self._contract(option_type, strike),
            spot=100,
            expiry_time=1,
            risk_free=0.05,
            div_yield=0,
            volatility=sigma,
        )
        return price(bs)

    def test_newton_fast_path_does_not_call_brent(self, monkeypatch) -> None:
        """A well-posed ATM round trip converges in the Newton fast path;
        the Brent fallback is never invoked."""
        import arbfree_vol.pricing.implied_vol as iv_mod

        calls = {"n": 0}
        real_brentq = iv_mod.brentq

        def _recording_brentq(f, a, b):
            calls["n"] += 1
            return real_brentq(f, a, b)

        monkeypatch.setattr(iv_mod, "brentq", _recording_brentq)

        target = self._target_price(OptionType.CALL, 100.0, 0.2)
        recovered = implied_vol(self._input(OptionType.CALL, target))

        assert recovered == approx(0.2, abs=1e-6)
        assert calls["n"] == 0, "Newton must converge without Brent"

    def test_brent_fallback_when_newton_fails(self, monkeypatch) -> None:
        """A deep-OTM call whose starting-point vega underflows makes the
        Newton path abort, so the Brent fallback must run and recover the
        true vol against a known reference."""
        import arbfree_vol.pricing.implied_vol as iv_mod

        calls = {"n": 0}
        real_brentq = iv_mod.brentq

        def _recording_brentq(f, a, b):
            calls["n"] += 1
            return real_brentq(f, a, b)

        monkeypatch.setattr(iv_mod, "brentq", _recording_brentq)

        target = self._target_price(OptionType.CALL, 200.0, 0.2)
        recovered = implied_vol(self._input(OptionType.CALL, target, strike=200.0))

        assert calls["n"] == 1, "the Brent fallback must run"
        assert recovered == approx(0.2, abs=1e-6)

    def test_vega_zero_branch_routes_to_brent(self, monkeypatch) -> None:
        """The ``v <= 0`` guard is a distinct Newton-abort branch: with
        vega patched to zero Newton cannot take a step and Brent must
        recover the true vol."""
        import arbfree_vol.pricing.implied_vol as iv_mod

        calls = {"n": 0}
        real_brentq = iv_mod.brentq

        def _recording_brentq(f, a, b):
            calls["n"] += 1
            return real_brentq(f, a, b)

        monkeypatch.setattr(iv_mod, "brentq", _recording_brentq)
        monkeypatch.setattr(iv_mod, "vega_floats", lambda *args, **kwargs: 0.0)

        target = self._target_price(OptionType.CALL, 100.0, 0.2)
        recovered = implied_vol(self._input(OptionType.CALL, target))

        assert calls["n"] == 1, "vega <= 0 must force the Brent fallback"
        assert recovered == approx(0.2, abs=1e-6)

    def test_custom_bounds_honored_on_brent_path(self) -> None:
        """On the Brent path ``low``/``high`` bracket the root: a root
        outside the bracket returns None, a root inside is returned."""
        target = self._target_price(OptionType.CALL, 200.0, 0.2)
        model = self._input(OptionType.CALL, target, strike=200.0)

        assert implied_vol(model, low=0.1, high=0.15) is None  # root 0.2 above high
        assert implied_vol(model, low=0.5, high=4.0) is None   # root 0.2 below low
        assert implied_vol(model, low=0.1, high=0.5) == approx(0.2, abs=1e-6)

    def test_newton_root_outside_custom_bounds_returns_none(self) -> None:
        """The ``[low, high]`` contract applies to the Newton fast path
        too: a converged Newton root above a custom ``high`` returns
        None, and a deep-OTM Newton root below a custom ``low`` returns
        None (no clamping, no silent out-of-bounds sigma)."""
        # True sigma 0.2 > custom high 0.15: Newton converges to ~0.2
        # (it is not bounded by high in the old sense), which is now
        # out of bounds -> None.
        target = self._target_price(OptionType.CALL, 100.0, 0.2)
        assert implied_vol(self._input(OptionType.CALL, target), high=0.15) is None

        # Deep OTM Newton root ~1.86e-16 < custom low 1e-4 -> None.
        target = self._target_price(OptionType.CALL, 500.0, 0.2)
        assert implied_vol(
            self._input(OptionType.CALL, target, strike=500.0), low=1e-4
        ) is None

    def test_invalid_prices(self) -> None:
        """Negative price and zero time-to-expiry are rejected at the input
        boundary (Pydantic ``gt=0``); unreachable prices return None."""
        with pytest.raises(ValidationError):
            self._input(OptionType.CALL, -1.0)
        with pytest.raises(ValidationError):
            self._input(OptionType.CALL, 5.0, T=0.0)

        # A call can never be worth the discounted spot (100.0); at and
        # beyond that level no root exists in the bracket.
        assert implied_vol(self._input(OptionType.CALL, 100.0)) is None
        assert implied_vol(self._input(OptionType.CALL, 200.0)) is None

    def test_edge_numerics(self) -> None:
        """Small and large sigma round-trip; extreme-strike prices are
        flat in sigma beyond double precision, so the solver honours its
        price-driven tolerance: a deep-ITM strike returns an in-bounds
        sigma that reproduces the price, while a deep-OTM strike whose
        Newton root lands below the low bound returns None (see the
        non-identifiability contract)."""
        for sigma in (0.02, 4.5):
            target = self._target_price(OptionType.CALL, 100.0, sigma)
            recovered = implied_vol(self._input(OptionType.CALL, target))
            assert recovered == approx(sigma, rel=1e-6), (
                f"sigma={sigma}: got {recovered}"
            )

        # Deep ITM: the solver must return an in-bounds sigma that
        # reproduces the target within the solver tolerance.
        target = self._target_price(OptionType.CALL, 5.0, 0.2)
        recovered = implied_vol(
            self._input(OptionType.CALL, target, strike=5.0)
        )
        assert recovered is not None
        assert 1e-6 <= recovered <= 5.0
        bs = BlackScholesInput(
            contract=self._contract(OptionType.CALL, 5.0),
            spot=100,
            expiry_time=1,
            risk_free=0.05,
            div_yield=0,
            volatility=recovered,
        )
        assert abs(price(bs) - target) < 1e-8, (
            f"strike=5.0: returned sigma {recovered} must reproduce "
            f"the target price within the solver tolerance"
        )

        # Deep OTM: the Newton root ~1.86e-16 sits below the default low
        # bound, so the contract returns None instead of an out-of-bounds
        # sigma.
        target = self._target_price(OptionType.CALL, 500.0, 0.2)
        assert implied_vol(
            self._input(OptionType.CALL, target, strike=500.0)
        ) is None

    def test_deep_otm_flat_price_root_below_low_returns_none(self) -> None:
        """Regression pin for the review finding: a deep-OTM call at
        K=500 priced at sigma=0.2 has a BS price of ~7.4e-15.  Newton's
        starting point ~1.86e-16 sits below the default low bound and
        the price is flat there (|price - target| < _NEWTON_TOL at
        sigma ~ 0), so the old solver returned 1.86e-16 — OUTSIDE
        [low, high].  The contract now returns None."""
        target = self._target_price(OptionType.CALL, 500.0, 0.2)
        recovered = implied_vol(self._input(OptionType.CALL, target, strike=500.0))
        assert recovered is None

    def test_deep_itm_flat_price_returns_in_bounds_sigma(self) -> None:
        """A deep-ITM call (K=5, spot=100) is flat in sigma: the price
        is intrinsic-dominated, so many sigmas reproduce it.  The solver
        must return a sigma inside [low, high] that reproduces the
        target within the solver price tolerance (the low endpoint
        1e-6 does exactly that)."""
        target = self._target_price(OptionType.CALL, 5.0, 0.2)
        recovered = implied_vol(self._input(OptionType.CALL, target, strike=5.0))

        assert recovered is not None
        assert 1e-6 <= recovered <= 5.0
        bs = BlackScholesInput(
            contract=self._contract(OptionType.CALL, 5.0),
            spot=100,
            expiry_time=1,
            risk_free=0.05,
            div_yield=0,
            volatility=recovered,
        )
        assert abs(price(bs) - target) < 1e-8

    def test_flat_price_region_sigmas_are_not_unique(self) -> None:
        """Pins the non-identifiability contract: in the deep-ITM K=5
        flat region, sigma=1e-6 and sigma=0.2 are materially different
        yet BOTH reproduce the target price within the solver's price
        tolerance.  This documents that the solver may return any
        in-bounds sigma that satisfies the price tolerance — non-
        uniqueness here is the documented flat-price behaviour, not a
        bug."""
        target = self._target_price(OptionType.CALL, 5.0, 0.2)

        for sigma in (1e-6, 0.2):
            bs = BlackScholesInput(
                contract=self._contract(OptionType.CALL, 5.0),
                spot=100,
                expiry_time=1,
                risk_free=0.05,
                div_yield=0,
                volatility=sigma,
            )
            assert abs(price(bs) - target) < 1e-8, (
                f"sigma={sigma} must reproduce the target within the "
                "solver price tolerance"
            )
