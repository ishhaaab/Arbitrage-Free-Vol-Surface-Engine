"""Violation heatmap and bar charts."""

from arbfree_vol.arbitrage.report import ArbitrageReport

from collections import Counter

from matplotlib.figure import Figure




def plot_violations_bar(report: ArbitrageReport) -> Figure:
    """Bar chart of violation counts grouped by type."""
    counts= Counter(v.kind.value for v in report.violations)

    fig= Figure(figsize=(8, 4))
    ax= fig.add_subplot(111)

    kinds= list(counts.keys())
    values= [counts[k] for k in kinds]
    colors= ["crimson", "darkorange", "goldenrod", "steelblue", "seagreen", "slategray"]

    ax.bar(kinds, values, color=colors[:len(kinds)])
    ax.set_xlabel("Violation type")
    ax.set_ylabel("Count")
    ax.set_title(f"Arbitrage violations (total: {len(report.violations)})")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return fig
