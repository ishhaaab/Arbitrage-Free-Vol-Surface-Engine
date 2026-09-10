"""Three figures used by the reproducible calibration study."""

from math import sqrt

import numpy as np
from matplotlib.figure import Figure

from arbfree_vol.report import CalibrationReport
from arbfree_vol.svi.model import svi_g, svi_total_variance


def plot_smiles(report: CalibrationReport) -> Figure:
    """Plot observations, raw SVI, and constrained SSVI by expiry."""
    constrained = {item.expiry_time: item for item in report.fitted_slices}
    baseline = {item.expiry_time: item for item in report.raw_svi_slices}
    maturities = sorted(set(constrained) | set(baseline))
    figure = Figure(figsize=(11, 3.2 * len(maturities)))
    for index, maturity in enumerate(maturities, start=1):
        axis = figure.add_subplot(len(maturities), 1, index)
        observed = constrained.get(maturity) or baseline[maturity]
        points = observed.data_points or ()
        if points:
            axis.scatter(*zip(*points), s=9, alpha=0.5, color="#555555", label="Observed")
        grid = np.linspace(report.certificate.k_min, report.certificate.k_max, 300)
        for label, fitted, color in (
            ("Raw SVI baseline", baseline.get(maturity), "#b04a3a"),
            ("Constrained SSVI", constrained.get(maturity), "#1f5a7a"),
        ):
            if fitted is None:
                continue
            p = fitted.params
            values = [svi_total_variance(float(k), p.a, p.b, p.rho, p.m, p.sigma) for k in grid]
            axis.plot(grid, values, color=color, linewidth=1.4, label=label)
        axis.set(title=f"T = {maturity:.3f} years", xlabel="log(K/F)", ylabel="total variance")
        axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def plot_surface(report: CalibrationReport) -> Figure:
    """Plot constrained SSVI implied volatility on the certificate grid."""
    ordered = sorted(report.fitted_slices, key=lambda item: item.expiry_time)
    grid = np.linspace(report.certificate.k_min, report.certificate.k_max, 241)
    maturities = [item.expiry_time for item in ordered]
    values = np.array(
        [
            [
                sqrt(variance / item.expiry_time) if variance >= 0 else float("nan")
                for k in grid
                for variance in [
                    svi_total_variance(float(k), p.a, p.b, p.rho, p.m, p.sigma)
                ]
            ]
            for item in ordered
            for p in [item.params]
        ]
    )
    figure = Figure(figsize=(10, 5))
    axis = figure.add_subplot(111)
    mesh = axis.pcolormesh(grid, maturities, values, shading="auto", cmap="viridis")
    figure.colorbar(mesh, ax=axis, label="implied volatility")
    axis.set(xlabel="log(K/F)", ylabel="maturity in years", title="Constrained SSVI surface")
    figure.tight_layout()
    return figure


def plot_constraints(report: CalibrationReport) -> Figure:
    """Plot grid minima for variance, butterfly density, and calendar spread."""
    ordered = sorted(report.fitted_slices, key=lambda item: item.expiry_time)
    if not ordered:
        raise ValueError("No constrained SSVI slices to plot")
    grid = np.linspace(report.certificate.k_min, report.certificate.k_max, report.certificate.grid_size)
    variance_minima: list[float] = []
    density_minima: list[float] = []
    calendar_minima: list[float] = []
    previous: np.ndarray | None = None
    for item in ordered:
        p = item.params
        variance = np.array([svi_total_variance(float(k), p.a, p.b, p.rho, p.m, p.sigma) for k in grid])
        density = np.array([svi_g(float(k), p.a, p.b, p.rho, p.m, p.sigma) for k in grid])
        variance_minima.append(float(variance.min()))
        density_minima.append(float(density.min()))
        if previous is not None:
            calendar_minima.append(float((variance - previous).min()))
        previous = variance

    figure = Figure(figsize=(10, 6))
    axes = figure.subplots(3, 1, sharex=False)
    maturities = [item.expiry_time for item in ordered]
    axes[0].plot(maturities, variance_minima, marker="o")
    axes[0].set_ylabel("min w")
    axes[1].plot(maturities, density_minima, marker="o")
    axes[1].set_ylabel("min g(k)")
    axes[2].plot(maturities[1:], calendar_minima, marker="o")
    axes[2].set(ylabel="min calendar margin", xlabel="later maturity")
    for axis in axes:
        axis.axhline(-report.certificate.tolerance, color="#b04a3a", linestyle="--", linewidth=1)
        axis.grid(alpha=0.2)
    figure.suptitle("Numerical certificate margins")
    figure.tight_layout()
    return figure
