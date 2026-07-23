from statistics import mean

from arbfree_vol.repair.report import RepairReport

from matplotlib.figure import Figure




def plot_comparison(
    raw_report: RepairReport,
    repaired_report: RepairReport,
) -> Figure:
    """Side by side comparison of raw and repaired surface metrics.

    Subplot 1: violation counts before vs after.
    Subplot 2: rejection rate, a single bar.
    Subplot 3: n_slices_fitted vs n_slices_input.
    """
    raw_m= raw_report.metrics
    rep_m= repaired_report.metrics

    fig= Figure(figsize=(12, 4))
    fig.suptitle("Repair comparison", fontsize=13)

    # 1: violation counts
    ax1= fig.add_subplot(131)
    ax1.bar(["Before", "After"], [raw_m.n_violations_before, rep_m.n_violations_after],
            color=["crimson", "seagreen"])
    ax1.set_ylabel("Violations")
    ax1.set_title("Violations")

    # 2: rejection rate
    ax2= fig.add_subplot(132)
    ax2.bar(["Rejected"], [rep_m.n_rejected],
            color="darkorange")
    ax2.set_ylabel("Quotes rejected")
    ax2.set_title(f"Rejection rate: {rep_m.rejection_rate:.1%}")

    # 3: slices
    ax3= fig.add_subplot(133)
    ax3.bar(["Input", "Fitted"], [rep_m.n_slices_input, rep_m.n_slices_fitted],
            color=["steelblue", "seagreen"])
    ax3.set_ylabel("Slices")
    ax3.set_title("Slices")

    fig.tight_layout()
    return fig


def plot_model_comparison(
    reports: dict[str, RepairReport],
    symbol: str = "SPY",
) -> Figure:
    """Bar chart comparing average RMSE and remaining violations across models.

    Parameters
    ----------
    reports:
        Mapping of model name → RepairReport, e.g.
        ``{"SVI": r_svi, "eSSVI": r_essvi, "SABR": r_sabr}``.
    symbol:
        Ticker symbol for the plot title.

    Returns
    -------
    Figure
    """
    model_names = list(reports.keys())
    n = len(model_names)
    colors = ["steelblue", "darkorange", "seagreen"]

    # Compute average RMSE per model
    avg_rmse: list[float] = []
    remaining_violations: list[int] = []
    for name in model_names:
        r = reports[name]
        rmses = [fs.rmse for fs in r.fitted_slices]
        avg_rmse.append(mean(rmses) if rmses else 0.0)
        remaining_violations.append(r.metrics.n_violations_after)

    fig = Figure(figsize=(10, 4))
    fig.suptitle(f"{symbol} model comparison", fontsize=13)

    # Subplot 1: average RMSE (log scale)
    ax1 = fig.add_subplot(121)
    bars1 = ax1.bar(model_names, avg_rmse, color=colors[:n])
    ax1.set_ylabel("Average RMSE (w-space)")
    ax1.set_title("Fit quality")
    ax1.set_yscale("log")
    for bar, val in zip(bars1, avg_rmse):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # Subplot 2: remaining violations
    ax2 = fig.add_subplot(122)
    bars2 = ax2.bar(model_names, remaining_violations, color=colors[:n])
    ax2.set_ylabel("Remaining violations")
    ax2.set_title("Arbitrage violations")
    for bar, val in zip(bars2, remaining_violations):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 str(val), ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return fig
