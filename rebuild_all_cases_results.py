"""
Rebuild the combined results file from per-case CSVs and optionally generate heatmaps.

Usage:
    python rebuild_all_cases_results.py
    python rebuild_all_cases_results.py --no-heatmaps
    python rebuild_all_cases_results.py --results-dir path/to/results
"""

import argparse
import logging
import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from config.constants import RESULTS_DIR, FIGURES_DIR
from evaluation.visualizations import Visualizer


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger("ResultsRebuilder")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Rebuild all_cases_results.csv from per-case results"
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing case*_results.csv files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: results-dir/all_cases_results.csv)"
    )
    parser.add_argument(
        "--no-heatmaps",
        action="store_true",
        help="Skip RMSE and R2 heatmap generation"
    )
    return parser.parse_args()


def find_case_files(results_dir: Path) -> List[Tuple[int, Path]]:
    """Find and sort per-case results files."""
    pattern = re.compile(r"case(\d+)_results\.csv$", re.IGNORECASE)
    case_files = []
    for path in results_dir.glob("case*_results.csv"):
        match = pattern.search(path.name)
        if match:
            case_num = int(match.group(1))
            case_files.append((case_num, path))
    return sorted(case_files, key=lambda item: item[0])


def rebuild_results(case_files: List[Tuple[int, Path]], logger: logging.Logger) -> pd.DataFrame:
    """Load per-case CSVs and combine into a single DataFrame."""
    combined = []
    for case_num, path in case_files:
        df = pd.read_csv(path)
        df["case"] = f"case{case_num}"
        combined.append(df)
        logger.info(f"Loaded {path.name} ({len(df)} rows)")

    if not combined:
        return pd.DataFrame()

    full_results = pd.concat(combined, ignore_index=True, sort=False)

    preferred_order = ["model", "MAE", "RMSE", "MAPE", "R2", "case", "RMSLE"]
    ordered_cols = [c for c in preferred_order if c in full_results.columns]
    ordered_cols += [c for c in full_results.columns if c not in ordered_cols]
    return full_results[ordered_cols]


def generate_heatmaps(full_results: pd.DataFrame, logger: logging.Logger) -> None:
    """Generate RMSE and R2 heatmaps from combined results."""
    visualizer = Visualizer(output_dir=FIGURES_DIR)

    if "RMSE" in full_results.columns:
        pivot_rmse = full_results.pivot_table(
            values="RMSE",
            index="model",
            columns="case",
            aggfunc="mean"
        )
        visualizer.plot_metrics_heatmap(
            metrics_df=pivot_rmse,
            title="Model Performance Comparison (RMSE)",
            filename="figure_7_10_model_ranking_heatmap",
            chapter="chapter7",
            metric_name="RMSE"
        )
        logger.info("Generated RMSE heatmap")
    else:
        logger.warning("RMSE column not found; skipping RMSE heatmap")

    if "R2" in full_results.columns:
        pivot_r2 = full_results.pivot_table(
            values="R2",
            index="model",
            columns="case",
            aggfunc="mean"
        )
        visualizer.plot_metrics_heatmap(
            metrics_df=pivot_r2,
            title="Model Performance Comparison (R2)",
            filename="figure_7_10_model_r2_heatmap",
            chapter="chapter7",
            metric_name="R2"
        )
        logger.info("Generated R2 heatmap")
    else:
        logger.warning("R2 column not found; skipping R2 heatmap")


def main() -> int:
    args = parse_args()
    logger = setup_logging()

    results_dir = args.results_dir
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return 1

    case_files = find_case_files(results_dir)
    if not case_files:
        logger.error(f"No case*_results.csv files found in: {results_dir}")
        return 1

    full_results = rebuild_results(case_files, logger)
    if full_results.empty:
        logger.error("No results loaded; nothing to write")
        return 1

    output_path = args.output or (results_dir / "all_cases_results.csv")
    full_results.to_csv(output_path, index=False)
    logger.info(f"Combined results saved to: {output_path}")

    if not args.no_heatmaps:
        generate_heatmaps(full_results, logger)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
