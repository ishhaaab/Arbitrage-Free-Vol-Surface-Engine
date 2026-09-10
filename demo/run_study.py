"""Run the frozen SPX calibration study and write its report and figures."""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arbfree_vol.calibration import calibrate_surface  # noqa: E402
from arbfree_vol.ingestion.loader import load_chain_csv  # noqa: E402
from arbfree_vol.plots import plot_constraints, plot_smiles, plot_surface  # noqa: E402


def _save(figure, path: Path) -> None:
    FigureCanvasAgg(figure).print_png(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "spx_2026-07-31.csv")
    parser.add_argument(
        "--metadata", type=Path, default=ROOT / "data" / "spx_2026-07-31_metadata.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "spx_2026-07-31")
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    surface, rejected = load_chain_csv(
        args.input,
        spot=float(metadata["spot"]),
        as_of=date.fromisoformat(metadata["as_of"]),
        risk_free=float(metadata["risk_free"]),
        div_yield=float(metadata["dividend_yield"]),
    )
    report = calibrate_surface(
        surface,
        rejected=rejected,
        dataset_source=str(metadata["source"]),
        dataset_timestamp=str(metadata["as_of"]),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    summary = asdict(report)
    summary.pop("cleaned_surface")
    summary.pop("rejected")
    summary.pop("raw_svi_slices")
    summary.pop("fitted_slices")
    summary.pop("fitted_ssvi_slices")
    summary["raw_svi_rmse"] = report.raw_svi_rmse
    summary["constrained_ssvi_rmse"] = report.constrained_ssvi_rmse
    summary["rejection_counts"] = report.rejection_counts
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    if report.fitted_slices:
        _save(plot_smiles(report), args.output / "smiles.png")
        _save(plot_surface(report), args.output / "surface.png")
        _save(plot_constraints(report), args.output / "constraints.png")
    print(json.dumps(summary["certificate"], indent=2))
    return 0 if report.certificate.certified else 2


if __name__ == "__main__":
    raise SystemExit(main())
