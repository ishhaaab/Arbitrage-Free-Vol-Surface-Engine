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

    def test_newton_root_outside_custom_bounds_falls_through_to_brent(self) -> None:
        """A Newton root outside a custom ``[low, high]`` is
        untrustworthy: the solver must neither return it nor return
        None — it falls through to the Brent search over the custom
        bracket.  ``None`` is returned only when the custom bracket
        itself contains no root."""
        # True sigma 0.2 > custom high 0.15: Newton converges to ~0.2
        # (it is not bounded by high during iteration), which is out of
        # the custom bracket -> Brent over [1e-6, 0.15] finds no root
        # (f(low) < 0 and f(0.15) < 0) -> None.
        target = self._target_price(OptionType.CALL, 100.0, 0.2)
        assert implied_vol(self._input(OptionType.CALL, target), high=0.15) is None

        # Deep OTM K=500: Newton's starting point ~1.86e-16 is below the
        # default low, so the Newton iterate is out of bounds; the true
        # root 0.2 is inside the DEFAULT bracket.  With a custom low
        # above the root (0.3), the Brent search over [0.3, 5] finds no
        # root -> None.  (With custom low=1e-4 the root stays inside the
        # custom bracket and the fall-through recovers it — see the
        # deep-OTM regression tests.)
        target = self._target_price(OptionType.CALL, 500.0, 0.2)
        assert implied_vol(
            self._input(OptionType.CALL, target, strike=500.0), low=0.3
        ) is None

    def test_no_root_when_target_outside_bracket_attainable_range(self) -> None:
        """A target price that no sigma in ``[low, high]`` can reach has
        no implied vol: a call priced above the maximum attainable price
        within the bracket (the price at sigma=high) — or below the
        minimum attainable (the price at sigma=low) — must return
        ``None``."""
        # Above the bracket maximum: call prices increase in sigma
        # (vega > 0), so price(high) is the largest value attainable in
        # the bracket; any larger target is unreachable.
        max_price = self._target_price(OptionType.CALL, 100.0, 5.0)
        assert implied_vol(
            self._input(OptionType.CALL, max_price + 0.5)
        ) is None

        # Below the bracket minimum: for the ATM call the intrinsic value
        # at sigma=low (~4.88) is the smallest attainable price in the
        # bracket; targets below it are unreachable.
        min_price = self._target_price(OptionType.CALL, 100.0, 1e-6)
        assert min_price > 1.0
        assert implied_vol(self._input(OptionType.CALL, 1.0)) is None
        assert implied_vol(self._input(OptionType.CALL, min_price - 0.1)) is None

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
        Newton root lands below the low bound falls through to Brent and
        returns the in-bounds root near the true sigma (see the
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

        # Deep OTM: Newton's starting point ~1.86e-16 sits below the
        # default low bound, so Newton's converged iterate is out of
        # bounds and the solver falls through to the bounded Brent
        # search, which recovers an in-bounds root near the true
        # sigma=0.2 (the BS price is flat there beyond double precision,
        # so the recovered root lies in the flat band rather than
        # exactly at 0.2).
        target = self._target_price(OptionType.CALL, 500.0, 0.2)
        recovered = implied_vol(
            self._input(OptionType.CALL, target, strike=500.0)
        )
        assert recovered is not None
        assert 1e-6 <= recovered <= 5.0
        # NON-IDENTIFIABILITY REPRODUCTION CHECK, not a solver-quality
        # assertion: at K=500 the BS price is flat at machine scale over
        # the whole root region (the vega contribution vanishes), so the
        # recovered root is ANY sigma in the flat band that satisfies the
        # price tolerance — "near 0.2" reproduces the documented
        # flat-price behaviour, it is NOT a tight-solver-quality bound.
        # The binding assertion is the price-reproduction tolerance below.
        assert recovered == approx(0.2, abs=0.05), (
            f"strike=500.0: expected a root near 0.2, got {recovered}"
        )
        bs = BlackScholesInput(
            contract=self._contract(OptionType.CALL, 500.0),
            spot=100,
            expiry_time=1,
            risk_free=0.05,
            div_yield=0,
            volatility=recovered,
        )
        assert abs(price(bs) - target) < 1e-8, (
            f"strike=500.0: returned sigma {recovered} must reproduce "
            f"the target price within the solver tolerance"
        )

    def test_deep_otm_falls_through_to_brent_and_recovers_root(self) -> None:
        """Regression pin for the over-correction: a deep-OTM call at
        K=500 priced at sigma=0.2 has a BS price of ~7.4e-15.  Newton's
        starting point ~1.86e-16 sits below the default low bound and
        the price is flat there (|price - target| < _NEWTON_TOL at
        sigma ~ 0), so the over-corrected solver returned None — a
        missed valid root.  The out-of-bounds Newton iterate is
        untrustworthy, so the solver must fall through to the bounded
        Brent search over [low, high] and recover the in-bounds root
        (~0.2; the BS price is flat beyond double precision in this
        region, so the recovered root lies in the flat band rather than
        exactly at 0.2) — never None and never the spurious 1.86e-16."""
        target = self._target_price(OptionType.CALL, 500.0, 0.2)
        recovered = implied_vol(self._input(OptionType.CALL, target, strike=500.0))

        assert recovered is not None
        assert 1e-6 <= recovered <= 5.0
        # NON-IDENTIFIABILITY REPRODUCTION CHECK (see the class docstring
        # and ``implied_vol``'s return contract): the deep-OTM price is
        # flat at machine scale in the root region, so the recovered root
        # lies anywhere in the flat band.  The ``approx(0.2, abs=0.05)``
        # assertion reproduces the documented non-unique-root behaviour —
        # it is NOT a tight solver-quality bound; the binding assertion
        # is the price-reproduction tolerance below.
        assert recovered == approx(0.2, abs=0.05), (
            f"expected the true IV ~0.2, got {recovered}"
        )
        bs = BlackScholesInput(
            contract=self._contract(OptionType.CALL, 500.0),
            spot=100,
            expiry_time=1,
            risk_free=0.05,
            div_yield=0,
            volatility=recovered,
        )
        assert abs(price(bs) - target) < 1e-8

    def test_deep_otm_custom_low_bound_recovers_root(self) -> None:
        """Deep-OTM recovery with a custom ``low`` bound: with
        ``low=1e-4`` the solver must return an in-bounds root (the true
        root ~0.1955 sits inside ``[1e-4, 5]``) — never None and never a
        boundary clamp — and the returned sigma must reproduce the target
        price within the documented price tolerance.

        The default ``low=1e-6`` case is covered by
        ``test_deep_otm_falls_through_to_brent_and_recovers_root``; this
        pins the same fall-through with a NON-default bound so a change
        that only honours the default bounds cannot silently pass."""
        target = self._target_price(OptionType.CALL, 500.0, 0.2)
        recovered = implied_vol(
            self._input(OptionType.CALL, target, strike=500.0), low=1e-4
        )

        assert recovered is not None
        assert 1e-4 <= recovered <= 5.0
        bs = BlackScholesInput(
            contract=self._contract(OptionType.CALL, 500.0),
            spot=100,
            expiry_time=1,
            risk_free=0.05,
            div_yield=0,
            volatility=recovered,
        )
        assert abs(price(bs) - target) < 1e-8, (
            f"strike=500.0 (low=1e-4): returned sigma {recovered} must "
            f"reproduce the target price within the solver tolerance"
        )

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
