"""Deep-dive: compute RMSE of degenerate (predecessor-copy) restart solutions
vs the unconstrained fit for the fallback slices.

This is a RESEARCH / DIAGNOSTIC tool, not part of the library test suite.
It fetches live SPY data, so all numbers are snapshot-in-time and vary day
to day.  It does NOT establish that the fallback slices are repairable —
it only compares how well the predecessor parameters and the unconstrained
fit explain the fallback slice's own data on this snapshot.
"""

import sys
from math import sqrt, log
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arbfree_vol.ssvi.model import SSVIParams, ssvi_w
from arbfree_vol.ssvi.calibration import fit_ssvi_slice
from arbfree_vol.ssvi.term_structure import fit_ssvi_surface_sequential
from arbfree_vol.variance import slice_total_variance
from arbfree_vol.ingestion.yfinance import fetch_chain
from arbfree_vol.forward import estimate_forward_curve, populate_per_slice_r


def fetch_and_extract():
    surface, _, _ = fetch_chain("SPY", max_expiries=20, min_T_years=7.0/365.0)
    fwd_curve = estimate_forward_curve(surface)
    populate_per_slice_r(surface, fwd_curve)
    slices_data = []
    for sl in sorted(surface.slices, key=lambda s: s.expiry_time):
        F = fwd_curve.get(sl.expiry_time)
        if F is None:
            continue
        strike_w = slice_total_variance(surface, sl)
        if len(strike_w) < 5:
            continue
        pts = [(log(strike / F), w) for strike, w in strike_w.items()]
        pts.sort()
        slices_data.append((sl.expiry_time, pts))
    return surface, slices_data


def rmse_of_params(params: SSVIParams, pts):
    ks = np.array([k for k, _ in points]) if (points := pts) else np.array([])
    ws = np.array([w for _, w in pts])
    preds = np.array([ssvi_w(k, params.theta, params.rho, params.psi) for k in ks])
    return float(np.sqrt(np.mean((preds - ws) ** 2)))


def main():
    print("Fetching SPY data...")
    surface, slices_data = fetch_and_extract()
    print(f"  Got {len(slices_data)} slices\n")

    result = fit_ssvi_surface_sequential(slices_data)
    sorted_fitted = sorted(result.fitted_slices, key=lambda x: x[0])
    data_by_T = {T: pts for T, pts in slices_data}

    print(f"Fallback slices: {[f'{T:.4f}' for T in result.fallback_slices]}\n")

    for fallback_T in result.fallback_slices:
        pts = data_by_T[fallback_T]

        # Unconstrained fit
        unc = fit_ssvi_slice(pts)
        unc_rmse = rmse_of_params(unc, pts)

        # Find predecessor
        prev_params = None
        prev_T = None
        for T_i, p_i in sorted_fitted:
            if T_i < fallback_T:
                prev_params = p_i
                prev_T = T_i
            else:
                break

        # Find successor
        next_params = None
        next_T = None
        for T_i, p_i in sorted_fitted:
            if T_i > fallback_T:
                next_params = p_i
                next_T = T_i
                break

        # RMSE if we use predecessor params on this slice's data
        prev_rmse = rmse_of_params(prev_params, pts) if prev_params else float("nan")

        # Find the "best" restart solution (if any converged)
        # We already know from the diagnostic that the restarts that converge
        # tend to just copy the predecessor. Let's confirm by trying with
        # the predecessor as warm-start.
        print(f"{'='*60}")
        print(f"T = {fallback_T:.4f}  (n_points = {len(pts)})")
        print(f"{'='*60}")
        print(f"  Unconstrained fit:  theta={unc.theta:.6f}  rho={unc.rho:.4f}  psi={unc.psi:.4f}  chi={unc.theta*unc.psi:.6f}  RMSE={unc_rmse:.8f}")
        if prev_params:
            print(f"  Predecessor ({prev_T:.4f}): theta={prev_params.theta:.6f}  rho={prev_params.rho:.4f}  psi={prev_params.psi:.4f}  chi={prev_params.theta*prev_params.psi:.6f}  RMSE={prev_rmse:.8f}")
            print(f"  theta_delta = {unc.theta - prev_params.theta:+.6f}  (NEGATIVE = violates H&M a)")
            print(f"  chi_delta   = {unc.theta*unc.psi - prev_params.theta*prev_params.psi:+.6f}")

            # What happens if we force theta = prev_theta + eps and refit?
            # We can't directly, but we can check: does the predecessor fit
            # this data well at all?
            print(f"\n  If we use predecessor params for this slice:")
            print(f"    RMSE = {prev_rmse:.8f}  (vs unconstrained = {unc_rmse:.8f})")
            print(f"    Ratio = {prev_rmse/unc_rmse:.1f}x worse" if unc_rmse > 0 else "    (unconstrained RMSE is 0)")

        if next_params:
            print(f"\n  Successor ({next_T:.4f}): theta={next_params.theta:.6f}  rho={next_params.rho:.4f}  psi={next_params.psi:.4f}  chi={next_params.theta*next_params.psi:.6f}")
            print(f"  unc->next theta_delta = {next_params.theta - unc.theta:+.6f}")

        # Check: what does the unconstrained fit's theta look like in the 
        # overall term structure?
        print(f"\n  ATM vol term structure around this slice:")
        for T_i, p_i in sorted_fitted:
            atm_w = p_i.theta
            atm_vol = sqrt(atm_w / T_i) if T_i > 0 else 0
            marker = " <-- FALLBACK" if T_i in result.fallback_slices else ""
            marker += " <-- THIS" if abs(T_i - fallback_T) < 1e-6 else ""
            print(f"    T={T_i:.4f}  theta={p_i.theta:.6f}  atm_vol={atm_vol:.4f}{marker}")

        print()

    # Overall summary: what is the theta term structure?
    print(f"\n{'='*60}")
    print("FULL THETA TERM STRUCTURE (fitted_slices = hard + fallback)")
    print(f"{'='*60}")
    for T_i, p_i in sorted_fitted:
        atm_vol = sqrt(p_i.theta / T_i) if T_i > 0 else 0
        is_fb = T_i in result.fallback_slices
        tag = " [FALLBACK]" if is_fb else ""
        print(f"  T={T_i:.4f}  theta={p_i.theta:.6f}  rho={p_i.rho:+.4f}  psi={p_i.psi:.4f}  chi={p_i.theta*p_i.psi:.6f}  atm_vol={atm_vol:.4f}{tag}")

    # Check: in the unconstrained fits, where does theta dip?
    print(f"\n{'='*60}")
    print("THETA DELTA ANALYSIS (consecutive pairs)")
    print(f"{'='*60}")
    for i in range(1, len(sorted_fitted)):
        T_prev, p_prev = sorted_fitted[i-1]
        T_curr, p_curr = sorted_fitted[i]
        d_theta = p_curr.theta - p_prev.theta
        d_chi = p_curr.theta * p_curr.psi - p_prev.theta * p_prev.psi
        marker = " THETA DIPS!" if d_theta < 0 else ""
        marker += " CHI DIPS!" if d_chi < 0 else ""
        print(f"  T={T_prev:.4f}->{T_curr:.4f}: d_theta={d_theta:+.6f}  d_chi={d_chi:+.6f}{marker}")


if __name__ == "__main__":
    main()
