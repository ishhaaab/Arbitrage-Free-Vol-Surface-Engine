"""Surface dynamics: PCA over a time series of fitted vol surfaces.

Shows how the fitted SVI surface moves through time.  We build a
series of surface snapshots, fit each one, stack the fitted parameters
into a matrix, run SVD-based PCA, and report the dominant deformation
modes (Level / Tilt / Curvature).

This runs on a deterministic synthetic time series (no network, identical
output every run) — the same input the repair pipeline would produce if
you collected daily option chains.  Collecting real daily snapshots is a
roadmap item (see ``docs/roadmap.md``, milestone 11).

Usage::

    python demo/dynamics_pca/dynamics_pca.py
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save to files, no GUI window

import matplotlib.pyplot as plt
import numpy as np

_OUT = Path(__file__).parent

parser = argparse.ArgumentParser(
    description="PCA over a time series of fitted SVI surfaces",
)
args = parser.parse_args()

_N_SNAPSHOTS = 30  # 30 business-ish days


# ---------------------------------------------------------------------------
# 1. Build the time series
# ---------------------------------------------------------------------------
def _synthetic_series() -> list[tuple[date, "object"]]:
    """30 snapshots of a VolSurface whose smile evolves.

    rho drifts from -0.5 toward -0.1 and sigma oscillates, so the PCA
    has real structure to find (a Level-like shift + a Tilt-like
    rotation).
    """
    from math import sqrt

    from arbfree_vol.models.option import OptionType
    from arbfree_vol.models.surface import VolSurface, ExpirySlice, Quote
    from arbfree_vol.svi.model import SVIParams, svi_total_variance
    from arbfree_vol.pricing.black_scholes import price_floats

    spot, r, q = 100.0, 0.05, 0.01
    exps = [0.25, 0.5, 1.0]
    base_ks = np.linspace(-0.35, 0.35, 11)

    def bs_price(otype, K, sigma, tt):
        return price_floats(spot, K, tt, r, q, sigma,
                            is_call=(otype == OptionType.CALL))

    surfaces = []
    for i in range(_N_SNAPSHOTS):
        t = i / (_N_SNAPSHOTS - 1)  # 0..1
        rho = -0.5 + 0.4 * t
        sigma_p = 0.10 + 0.06 * np.sin(2 * np.pi * t)
        slices = []
        for T in exps:
            p = SVIParams(a=0.015 + 0.01 * T, b=0.32, rho=rho,
                          m=0.0, sigma=sigma_p + 0.02 * T)
            quotes = []
            for k in base_ks:
                w = svi_total_variance(float(k), p.a, p.b, p.rho, p.m, p.sigma)
                sigma = sqrt(w / T)
                for otype in (OptionType.CALL, OptionType.PUT):
                    K = float(spot * np.exp(k))
                    price = bs_price(otype, K, sigma, T)
                    quotes.append(Quote(strike=K, option_type=otype, price=price))
            slices.append(ExpirySlice(expiry_time=T, quotes=quotes))
        surfaces.append((
            date(2030, 1, 1) + timedelta(days=i),
            VolSurface(spot=spot, risk_free=r, div_yield=q, slices=slices),
        ))
    return surfaces


# ---------------------------------------------------------------------------
# 2. Fit, stack, PCA
# ---------------------------------------------------------------------------
def main() -> None:
    print()
    print("=" * 62)
    print("   Surface dynamics: PCA over fitted vol surfaces")
    print("   SYNTHETIC time series (no network)")
    print("=" * 62)

    # Build the series (date, VolSurface) pairs
    pairs = _synthetic_series()

    from arbfree_vol.dynamics import (
        fit_surface_series, parameter_matrix, pca_deformations,
        principal_mode_labels,
    )
    from arbfree_vol.surface.interpolate import iv_at

    # Fit each snapshot through the repair pipeline (SVI), then stack.
    series = fit_surface_series(pairs)
    # Map snapshot date -> original surface (for spot/r/q in the plots).
    surf_by_date = {d: s for d, s in pairs}
    print(f"\n  Fitted {len(series.snapshots)} snapshots "
          f"({series.snapshots[0].snapshot_date} -> "
          f"{series.snapshots[-1].snapshot_date})")

    matrix, buckets, labels = parameter_matrix(series)
    print(f"  Parameter matrix: {matrix.shape[0]} snapshots x "
          f"{matrix.shape[1]} features")
    print(f"  Expiry buckets: {[round(b, 3) for b in buckets]}")

    result = pca_deformations(matrix, n_components=3)
    print("\n  Explained variance by principal component:")
    for i, ev in enumerate(result.explained_variance_ratio):
        label = principal_mode_labels(i + 1)[-1]
        cumulative = sum(result.explained_variance_ratio[:i + 1])
        print(f"    PC{i + 1} ({label:<10}): {ev:>7.1%}  "
              f"(cumulative {cumulative:>7.1%})")

    # Plot 1 — explained variance + mode labels
    fig, ax = plt.subplots(figsize=(8, 4))
    evs = result.explained_variance_ratio
    xs = np.arange(1, len(evs) + 1)
    ax.bar(xs, evs, color="#1f77b4")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"PC{i} ({principal_mode_labels(i)[-1]})"
                        for i in xs])
    ax.set_ylabel("Explained variance")
    ax.set_title("Synthetic surface dynamics: PCA explained variance")
    fig.tight_layout()
    fig.savefig(str(_OUT / "dynamics_pca_variance.png"), dpi=150)
    plt.close(fig)
    print("\nSaved: dynamics_pca_variance.png")

    # Plot 2 — component loadings across expiry/parameter features.
    # Only the top feature-loading columns per component are shown.
    fig, axes = plt.subplots(len(evs), 1, figsize=(10, 3.2 * len(evs)),
                             sharex=True)
    if len(evs) == 1:
        axes = [axes]
    for i, comp in enumerate(result.components):
        ax = axes[i]
        loadings = comp
        order = np.argsort(-np.abs(loadings))
        top = order[: min(8, len(order))]
        feature_labels = [labels[j].replace("_", " ") for j in top]
        ax.bar(range(len(top)), loadings[top], color="#ff7f0e")
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(feature_labels, rotation=30, ha="right")
        ax.set_ylabel("Loading")
        ax.set_title(f"PC{i + 1} ({principal_mode_labels(i + 1)[-1]}): "
                     f"top loadings")
    fig.tight_layout()
    fig.savefig(str(_OUT / "dynamics_pca_loadings.png"), dpi=150)
    plt.close(fig)
    print("Saved: dynamics_pca_loadings.png")

    # Plot 3 — ATM IV term structure across the series (how the surface
    # shifts through time).
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, sn in enumerate(series.snapshots):
        surf = surf_by_date[sn.snapshot_date]
        fs = rebuild_surface(sn, surf)
        ts, ivs = [], []
        for sl in sn.fitted_slices:
            try:
                ts.append(sl.expiry_time * 365.25)
                ivs.append(iv_at(fs, sl.forward_price, sl.expiry_time))
            except (ValueError, ZeroDivisionError):
                continue
        alpha = 0.25 + 0.75 * (i / len(series.snapshots))
        ax.plot(ts, ivs, color="#2ca02c", alpha=alpha, linewidth=0.8)
    ax.set_xlabel("Expiry (days)")
    ax.set_ylabel("ATM implied vol")
    ax.set_title("ATM term structure through time (synthetic); darker = later")
    fig.tight_layout()
    fig.savefig(str(_OUT / "dynamics_pca_term_structure.png"), dpi=150)
    plt.close(fig)
    print("Saved: dynamics_pca_term_structure.png")


def rebuild_surface(snapshot, original_surface):
    """Rebuild a FittedSurface from a snapshot's fitted slices + the
    original surface's spot/rates (so iv_at works without hardcoding)."""
    from arbfree_vol.models.fitted import FittedSurface
    return FittedSurface(
        spot=original_surface.spot,
        risk_free=original_surface.risk_free,
        div_yield=original_surface.div_yield,
        forward_curve=tuple((sl.expiry_time, sl.forward_price)
                            for sl in snapshot.fitted_slices),
        fitted_slices=snapshot.fitted_slices,
    )


if __name__ == "__main__":
    main()
