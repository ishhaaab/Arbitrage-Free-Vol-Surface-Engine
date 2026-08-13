"""Shared ground-truth case schema.

``GroundTruthCase`` is a frozen dataclass carrying the hand-derived analytic
label for one model configuration plus the proof of that label.  Nothing in
this module may call the repo's detectors / verifiers to DERIVE a label: the
label fields are analytic literals computed by hand in the case files.

The ``source`` field must never claim "published paper example" unless the
numbers literally come from a published source that can be cited.  All case
numbers in this module are self-derived from the analytic conditions; the
``PAPER_EXAMPLES`` registry below is the clearly-marked (currently empty)
slot where GJ2014 / HM2019 worked examples can be pasted with citations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from arbfree_vol.models.option import OptionType
from arbfree_vol.models.surface import Quote, ExpirySlice, VolSurface
from arbfree_vol.pricing.black_scholes import price_floats
from arbfree_vol.sabr.model import SABRParams
from arbfree_vol.ssvi.model import SSVIParams, eSSVISurfaceParams, ssvi_w
from arbfree_vol.svi.model import SVIParams

KnownLabel = Literal[
    "arb_free",
    "butterfly_violation",
    "calendar_violation",
    "both_violation",
    "boundary",
    "unknown_by_design",
]
# ``unknown_by_design`` is RESERVED for future, not-yet-hand-derived cases: it
# is the ``known_label`` default precisely so a half-built case is loudly
# rejected until its real label is filled in.  No shipped ``GroundTruthCase``
# may carry it — the ground-truth tests reject it explicitly.


@dataclass(frozen=True)
class GroundTruthCase:
    """One hand-derived ground-truth case.

    Parameters
    ----------
    name : str
        Unique case name (used as a test id).
    model : {"svi", "essvi", "sabr"}
        Which parametrisation the case exercises.
    params : SVIParams | SSVIParams | eSSVISurfaceParams | SABRParams | None
        The typed params object the case is built from.  For multi-slice
        eSSVI surface cases this is the surface-level object (None if the
        case only carries per-slice params); the per-slice objects live in
        ``slices``.
    slices : tuple[SSVIParams, ...]
        For eSSVI surface cases: the per-expiry SSVIParams ordered by
        increasing maturity (exactly what ``verify_hm_condition`` consumes).
        Empty for single-slice / raw-SVI / SABR cases.
    known_label : KnownLabel
        The HAND-DERIVED label (analytic conditions, never repo calls).
        ``"unknown_by_design"`` is reserved for future cases and must never
        appear on a shipped case (the ground-truth tests reject it).
    expected_hm_condition : bool | None
        Hand-derived Hendriks-Martini Prop 3.1 expectation, or None when not
        applicable (SVI / SABR cases have no native SSVI params).
    expected_grid_calendar_free : bool | None
        Hand-derived expectation for the native-SSVI grid calendar check, or
        None when not applicable.
    source : str
        Derivation basis ("self-derived (GJ 2014 Theorem 4.2 + Lemma 4.2
        conditions)", "self-derived (Gatheral 2004 Eq 1.10)", ...).
    proof_note : str
        The analytic reasoning, e.g.
        ``"theta*psi*(1+|rho|) = 2.0*2.0*1.0 = 4.0 exactly; ..."``.
    """

    name: str
    model: Literal["svi", "essvi", "sabr"]
    params: Any = None
    slices: tuple[SSVIParams, ...] = ()
    surface_params: eSSVISurfaceParams | None = None
    known_label: KnownLabel = "unknown_by_design"
    expected_hm_condition: bool | None = None
    expected_grid_calendar_free: bool | None = None
    source: str = ""
    proof_note: str = ""


# ---------------------------------------------------------------------------
# Published worked examples registry (deliberately empty)
# ---------------------------------------------------------------------------
# GJ2014 / HM2019 worked examples can be pasted here WITH citation, e.g.:
#
#   "essvi_gj2014_example": GroundTruthCase(
#       name="essvi_gj2014_example",
#       model="essvi",
#       ...
#       source="Gatheral & Jacquier (2014) Example X.Y ...",
#   )
#
# Until such an entry exists the registry stays empty; every case shipped in
# this module is self-derived (see the ``source`` field of each case).
PAPER_EXAMPLES: dict[str, GroundTruthCase] = {}


# ---------------------------------------------------------------------------
# Shared synthetic-market helpers (used by the repair-path and fit-quality
# tests).  Quotes are generated from a known smile via Black-Scholes, so the
# ground truth (the smile that produced them) is known by construction.
# ---------------------------------------------------------------------------


def bs_quote(
    strike: float,
    expiry_time: float,
    sigma: float,
    *,
    spot: float = 100.0,
    risk_free: float = 0.05,
    div_yield: float = 0.01,
    half_spread: float = 0.005,
) -> list[Quote]:
    """Call+put quotes at one strike priced from ``sigma``.

    The mid price is the Black-Scholes price at ``sigma``; bid/ask sit
    ``half_spread`` either side of the mid (a small, documented market
    spread).  Prices are floored at a tiny positive value so the repo's
    ``ImpliedVolInput`` (``market_price > 0``) never rejects a deep-OTM
    quote.
    """
    call = max(price_floats(spot, strike, expiry_time, risk_free, div_yield,
                            sigma, True), 1e-8)
    put = max(price_floats(spot, strike, expiry_time, risk_free, div_yield,
                           sigma, False), 1e-8)
    return [
        Quote(strike=strike, option_type=OptionType.CALL, price=call,
              bid=call * (1.0 - half_spread), ask=call * (1.0 + half_spread)),
        Quote(strike=strike, option_type=OptionType.PUT, price=put,
              bid=put * (1.0 - half_spread), ask=put * (1.0 + half_spread)),
    ]


def build_essvi_quote_surface(
    slices: list[tuple[float, SSVIParams]],
    *,
    n_k: int = 13,
    k_lo: float = -0.6,
    k_hi: float = 0.6,
    spot: float = 100.0,
    risk_free: float = 0.05,
    div_yield: float = 0.01,
) -> VolSurface:
    """Build a quote ``VolSurface`` whose IV smile is the given eSSVI slices.

    Each slice's quotes sit at ``K = F(T)*exp(k)`` for ``k`` on a uniform
    grid over ``[k_lo, k_hi]``; the implied vol at each strike is
    ``sqrt(ssvi_w(k, theta, rho, psi) / T)``.  This is the surface the
    repair-path tests feed to ``repair(use_ssvi=True)``.
    """
    from math import exp, sqrt

    surface_slices: list[ExpirySlice] = []
    for T, params in slices:
        F = spot * exp((risk_free - div_yield) * T)
        quotes: list[Quote] = []
        for k in (k_lo + (k_hi - k_lo) * i / (n_k - 1) for i in range(n_k)):
            w = ssvi_w(float(k), params.theta, params.rho, params.psi)
            sigma = sqrt(w / T)
            quotes.extend(bs_quote(F * exp(k), T, sigma, spot=spot,
                                   risk_free=risk_free, div_yield=div_yield))
        surface_slices.append(ExpirySlice(expiry_time=T, quotes=quotes))
    return VolSurface(spot=spot, risk_free=risk_free, div_yield=div_yield,
                      slices=surface_slices)


def build_svi_quote_surface(
    T: float,
    params: SVIParams,
    *,
    n_k: int = 13,
    k_lo: float = -0.6,
    k_hi: float = 0.6,
    spot: float = 100.0,
    risk_free: float = 0.05,
    div_yield: float = 0.01,
) -> VolSurface:
    """Build a one-slice quote surface whose IV smile is a raw-SVI smile."""
    from math import exp, sqrt

    from arbfree_vol.svi.model import svi_total_variance

    F = spot * exp((risk_free - div_yield) * T)
    quotes: list[Quote] = []
    for k in (k_lo + (k_hi - k_lo) * i / (n_k - 1) for i in range(n_k)):
        w = svi_total_variance(float(k), params.a, params.b, params.rho,
                               params.m, params.sigma)
        sigma = sqrt(w / T)
        quotes.extend(bs_quote(F * exp(k), T, sigma, spot=spot,
                               risk_free=risk_free, div_yield=div_yield))
    return VolSurface(spot=spot, risk_free=risk_free, div_yield=div_yield,
                      slices=[ExpirySlice(expiry_time=T, quotes=quotes)])
