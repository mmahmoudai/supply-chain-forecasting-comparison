"""Generate a full documentation report for the 4-case model comparison."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

from config.constants import (
    CASE1_PATH,
    CASE2_TRAIN_PATH,
    CASE2_TEST_PATH,
    CASE3_PATH,
    CASE4_PATH,
    OUTPUTS_DIR,
    TARGET_COLUMNS
)
from new_data_insights import load_csv, summarize_case, render_case_section


REPORT_PATH = OUTPUTS_DIR / "reports" / "full_documentation_report.md"


def format_table(df: pd.DataFrame) -> List[str]:
    """Render a DataFrame as a Markdown table."""
    if df.empty:
        return ["- No results available."]

    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |"
    ]

    for _, row in df.iterrows():
        values = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                values.append("")
            elif isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")

    return lines


def list_figures(fig_dir: Path, header: str) -> List[str]:
    """List figure filenames in a directory."""
    lines = [header, ""]
    if not fig_dir.exists():
        lines.append("- No figures directory found.")
        lines.append("")
        return lines

    figures = sorted(fig_dir.glob("*.png"))
    if not figures:
        lines.append("- No figures found.")
        lines.append("")
        return lines

    for fig in figures:
        lines.append(f"- {fig.name}")
    lines.append("")
    return lines


def main() -> None:
    lines: List[str] = []
    lines.append("# Full Model Comparison Documentation")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    lines.append("## DATASET AVAILABILITY")
    lines.append("")
    lines.append(f"- Case 1 path: {CASE1_PATH}")
    lines.append(f"- Case 2 train path: {CASE2_TRAIN_PATH}")
    lines.append(f"- Case 2 test path: {CASE2_TEST_PATH} (exists={CASE2_TEST_PATH.exists()})")
    lines.append(f"- Case 3 path: {CASE3_PATH}")
    lines.append(f"- Case 4 path: {CASE4_PATH}")
    lines.append("")

    lines.append("## EVALUATION NOTES")
    lines.append("")
    lines.append("- Time-series models use aggregated daily or weekly series.")
    lines.append("- Tabular models use row-level features when available.")
    lines.append("- For cases 2 and 3, time-series vs tabular metrics are not directly comparable.")
    lines.append("")

    # Case summaries
    summaries = []
    summaries.append(summarize_case("case1", load_csv(CASE1_PATH), TARGET_COLUMNS["case1"]))
    summaries.append(summarize_case("case2", load_csv(CASE2_TRAIN_PATH), TARGET_COLUMNS["case2"]))
    summaries.append(summarize_case("case3", load_csv(CASE3_PATH, encoding="latin-1"), TARGET_COLUMNS["case3"]))
    summaries.append(summarize_case("case4", load_csv(CASE4_PATH), TARGET_COLUMNS["case4"]))

    lines.append("## CASE SUMMARIES")
    lines.append("")
    for summary in summaries:
        lines.extend(render_case_section(summary))

    # Model coverage and results
    results_path = OUTPUTS_DIR / "results" / "all_cases_results.csv"
    if results_path.exists():
        results_df = pd.read_csv(results_path)
        if not results_df.empty:
            lines.append("## MODEL COVERAGE")
            lines.append("")
            models = sorted(results_df["model"].unique())
            lines.append(f"- Models: {', '.join(models)}")
            cases = sorted(results_df["case"].unique())
            for case in cases:
                case_models = sorted(results_df[results_df["case"] == case]["model"].unique())
                lines.append(f"- {case} models: {', '.join(case_models)}")
                missing = [m for m in models if m not in case_models]
                if missing:
                    lines.append(f"- {case} missing: {', '.join(missing)}")
            lines.append("")

            lines.append("## FULL RESULTS TABLE")
            lines.append("")
            table_df = results_df.copy()
            table_df = table_df.sort_values(["case", "RMSE"])
            cols = [c for c in ["case", "model", "MAE", "RMSE", "R2", "MAPE", "RMSLE"] if c in table_df.columns]
            lines.extend(format_table(table_df[cols]))
            lines.append("")
        else:
            lines.append("## MODEL RESULTS")
            lines.append("")
            lines.append("- Results file is empty.")
            lines.append("")
    else:
        lines.append("## MODEL RESULTS")
        lines.append("")
        lines.append("- Results file not found.")
        lines.append("")

    # Figures
    lines.extend(list_figures(OUTPUTS_DIR / "figures" / "chapter6", "## FIGURES (CHAPTER 6)"))
    lines.extend(list_figures(OUTPUTS_DIR / "figures" / "chapter7", "## FIGURES (CHAPTER 7)"))

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
