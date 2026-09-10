"""Three figures used by the reproducible calibration study."""

import numpy as np
from matplotlib.figure import Figure

from arbfree_vol.report import CalibrationReport
from arbfree_vol.svi.model import svi_g, svi_total_variance


def plot_smiles(report: CalibrationReport) -> Figure:
    """Plot observations, raw SVI, and constrained SSVI by expiry."""
    constrained = {item.expiry_time: item for item in report.fitted_slices}
    baseline = {item.expiry_time: item for item in report.raw_svi_slices}
    maturities = sorted(set(constrained) | set(baseline))
    columns = 2
    rows = (len(maturities) + columns - 1) // columns
    figure = Figure(figsize=(12, 3.2 * rows))
    for index, maturity in enumerate(maturities, start=1):
        axis = figure.add_subplot(rows, columns, index)
        observed = constrained.get(maturity) or baseline[maturity]
        points = observed.data_points or ()
        if points:
            axis.scatter(*zip(*points), s=10, alpha=0.45, color="#555555", label="Observed")
            observed_k = [point[0] for point in points]
            padding = max(0.04, 0.08 * (max(observed_k) - min(observed_k)))
            k_min = max(report.certificate.k_min, min(observed_k) - padding)
            k_max = min(report.certificate.k_max, max(observed_k) + padding)
        else:
            k_min = report.certificate.k_min
            k_max = report.certificate.k_max
        grid = np.linspace(k_min, k_max, 300)
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
        axis.grid(alpha=0.15)
    for index in range(len(maturities) + 1, rows * columns + 1):
        figure.add_subplot(rows, columns, index).set_visible(False)
    handles, labels = figure.axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=3,
        frameon=False,
    )
    figure.suptitle("Observed smiles and calibrated fits", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    return figure


def plot_surface(report: CalibrationReport) -> Figure:
    """Plot constrained SSVI implied volatility on the certificate grid."""
    ordered = sorted(report.fitted_slices, key=lambda item: item.expiry_time)
    grid = np.linspace(report.certificate.k_min, report.certificate.k_max, 241)
    slice_maturities = np.array([item.expiry_time for item in ordered])
    slice_variances = np.array(
        [
            [
                variance
                for k in grid
                for variance in [
                    svi_total_variance(float(k), p.a, p.b, p.rho, p.m, p.sigma)
                ]
            ]
            for item in ordered
            for p in [item.params]
        ]
    )
    maturities = np.linspace(slice_maturities[0], slice_maturities[-1], 180)
    variances = np.array(
        [np.interp(maturities, slice_maturities, column) for column in slice_variances.T]
    ).T
    values = np.sqrt(np.maximum(variances, 0.0) / maturities[:, None])
    figure = Figure(figsize=(10, 5.4))
    axis = figure.add_subplot(111)
    mesh = axis.contourf(grid, maturities, values, levels=30, cmap="viridis")
    figure.colorbar(mesh, ax=axis, label="implied volatility")
    axis.set(
        xlabel="log(K/F)",
        ylabel="maturity in years",
        title="Constrained SSVI implied-volatility surface",
    )
    axis.scatter(
        np.zeros_like(slice_maturities),
        slice_maturities,
        marker="|",
        color="white",
        alpha=0.8,
        label="calibrated expiries",
    )
    axis.legend(loc="upper right", frameon=False, labelcolor="white")
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
