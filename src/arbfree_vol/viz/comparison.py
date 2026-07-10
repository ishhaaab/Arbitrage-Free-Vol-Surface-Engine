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
