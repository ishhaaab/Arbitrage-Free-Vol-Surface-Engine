
from arbfree_vol.models.surface import VolSurface
from arbfree_vol.repair.report import RepairReport
from arbfree_vol.repair.engine import repair


def iterative_repair(surface: VolSurface, 
                     max_iters: int= 5) -> list[RepairReport]:
    """Run the repair pipeline up to *max_iters* times.

    Each iteration:
      1. Detects quote level violations on the current surface.
      2. Rejects offending quotes and builds a cleaned surface.
      3. Estimates the forward curve from survivors and refits SVI.
      4. Checks the fitted surface for remaining violations.

    The next iteration runs on the cleaned surface from the previous
    step.  The loop stops when:
      - No remaining violations are found (converged), or
      - The cleaned surface becomes None (no survivors), or
      - *max_iters* is reached.

    Returns a list of RepairReport objects (one per iteration).
    The final element is the last repair result.
    """
    reports: list[RepairReport]= []
    current= surface

    for _ in range(max_iters):
        report= repair(current)
        reports.append(report)

        # converged, ie no violations remain
        if report.remaining_violations.is_arbitrage_free:
            break

        # cleaned surface is gone, ie nothing left to repair
        if report.cleaned_surface is None:
            break

        # surface hasn't changed (converged without reaching 0)
        if (
            len(reports) >= 2
            and report.metrics.n_rejected== 0
            and reports[-2].metrics.n_rejected== 0
        ):
            break

        current= report.cleaned_surface

    return reports
