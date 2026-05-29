"""Data insights report for new_dataset cases with feature actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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


REPORT_PATH = OUTPUTS_DIR / "reports" / "new_data_insights_report.md"


@dataclass
class CaseSummary:
    name: str
    rows: int
    cols: int
    missing_total: int
    missing_pct: float
    target_stats: Dict[str, float]
    categorical_cardinality: List[Tuple[str, int]]
    numeric_correlations: List[Tuple[str, float]]
    leakage_flags: List[str]


FEATURE_ACTIONS = {
    "case1": {
        "added": [
            "lag_1", "lag_7", "lag_14",
            "rolling_mean_7", "rolling_std_7",
            "rolling_mean_14", "rolling_std_14",
            "calendar features (month, day_of_week, week_of_year, quarter, is_weekend, is_month_start, is_month_end)",
            "time_index",
            "month_sin", "month_cos",
            "week_of_year_sin", "week_of_year_cos",
            "day_of_week_sin", "day_of_week_cos",
            "quarter_sin", "quarter_cos"
        ],
        "dropped": [],
        "leakage": []
    },
    "case2": {
        "added": [
            "price_ratio",
            "log_checkout_price", "log_base_price",
            "discount_amount", "discount_percent", "discount_yn",
            "price_change", "price_change_yn", "price_change_pct",
            "lag_1", "lag_4", "lag_8",
            "rolling_mean_4", "rolling_std_4",
            "quarter", "year", "week_in_year",
            "week_in_year_sin", "week_in_year_cos",
            "quarter_sin", "quarter_cos",
            "promo_count", "both_promo", "promo_discount_interaction",
            "Region/center_type/category/cuisine one-hot",
            "center_id one-hot",
            "meal_id one-hot"
        ],
        "dropped": ["id", "prev_checkout_price"],
        "leakage": []
    },
    "case3": {
        "added": [
            "Order_Month", "Order_DayOfWeek", "Order_WeekOfYear",
            "order_month_sin", "order_month_cos",
            "order_dayofweek_sin", "order_dayofweek_cos",
            "order_weekofyear_sin", "order_weekofyear_cos",
            "shipping_mode_rank", "is_expedited",
            "order_value_per_unit",
            "sales_per_item",
            "discount_per_item",
            "discount_to_total",
            "sales_to_total",
            "log_order_item_total",
            "log_sales",
            "log_product_price",
            "log_order_item_discount",
            "Shipping Mode/Type/Market/Order Region one-hot",
            "Order Country/Order State target encoding"
        ],
        "dropped": [
            "Delivery Status",
            "Late_delivery_risk",
            "shipping date (DateOrders)",
            "Order Profit Per Order",
            "Order Item Profit Ratio",
            "Customer identifiers"
        ],
        "leakage": [
            "Delivery Status",
            "Late_delivery_risk",
            "shipping date (DateOrders)"
        ]
    },
    "case4": {
        "added": [
            "lag_1", "lag_7",
            "rolling_mean_7", "rolling_std_7",
            "price_ratio",
            "price_gap",
            "price_after_discount",
            "discount_amount",
            "inventory_to_order_ratio",
            "inventory_buffer",
            "log_price",
            "log_competitor_pricing",
            "log_inventory_level",
            "log_units_ordered",
            "calendar features (month, day_of_week, week_of_year, quarter, is_weekend, is_month_start, is_month_end)",
            "time_index",
            "month_sin", "month_cos",
            "week_of_year_sin", "week_of_year_cos",
            "day_of_week_sin", "day_of_week_cos",
            "quarter_sin", "quarter_cos",
            "Category/Region/Weather Condition/Seasonality one-hot",
            "Store ID one-hot",
            "Product ID one-hot"
        ],
        "dropped": ["Demand Forecast"],
        "leakage": ["Demand Forecast (correlation > 0.95)"]
    }
}


def load_csv(path: Path, encoding: str | None = None) -> pd.DataFrame:
    if encoding:
        return pd.read_csv(path, encoding=encoding)
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns=lambda c: c.replace("\ufeff", "").replace("ï»¿", ""))
    return df


def summarize_case(name: str, df: pd.DataFrame, target_col: str) -> CaseSummary:
    df = normalize_columns(df)
    rows, cols = df.shape
    missing_total = int(df.isna().sum().sum())
    missing_pct = (missing_total / (rows * cols)) * 100 if rows and cols else 0.0

    target_series = pd.to_numeric(df[target_col], errors="coerce") if target_col in df.columns else pd.Series([])
    target_stats = {}
    if not target_series.empty:
        target_stats = {
            "min": float(np.nanmin(target_series)),
            "max": float(np.nanmax(target_series)),
            "mean": float(np.nanmean(target_series)),
            "median": float(np.nanmedian(target_series)),
            "std": float(np.nanstd(target_series)),
            "zeros": float((target_series == 0).sum()),
            "negatives": float((target_series < 0).sum())
        }

    categorical_cols = [c for c in df.columns if df[c].dtype == "object"]
    categorical_cardinality = [(c, int(df[c].nunique())) for c in categorical_cols]
    categorical_cardinality.sort(key=lambda x: x[1], reverse=True)

    numeric_df = df.select_dtypes(include=["number"]).copy()
    if target_col not in numeric_df.columns and target_col in df.columns:
        numeric_df[target_col] = target_series
    if target_col in numeric_df.columns:
        corr = numeric_df.dropna(subset=[target_col]).corrwith(numeric_df[target_col])
        corr = corr.drop(labels=[target_col], errors="ignore").dropna()
        corr = corr.sort_values(key=lambda s: s.abs(), ascending=False)
        numeric_correlations = [(k, float(v)) for k, v in corr.head(8).items()]
    else:
        numeric_correlations = []

    leakage_flags = []
    if name == "case3":
        leak_cols = ["Delivery Status", "Late_delivery_risk", "shipping date (DateOrders)"]
        leakage_flags = [c for c in leak_cols if c in df.columns]
    if name == "case4" and target_col in df.columns:
        if target_col in numeric_df.columns:
            high_corr = corr[abs(corr) > 0.95] if "corr" in locals() else pd.Series(dtype=float)
            leakage_flags.extend([f"{c} (corr={high_corr[c]:.3f})" for c in high_corr.index])

    return CaseSummary(
        name=name,
        rows=rows,
        cols=cols,
        missing_total=missing_total,
        missing_pct=missing_pct,
        target_stats=target_stats,
        categorical_cardinality=categorical_cardinality[:8],
        numeric_correlations=numeric_correlations,
        leakage_flags=leakage_flags
    )


def format_stats(stats: Dict[str, float]) -> str:
    if not stats:
        return "n/a"
    return (
        f"min={stats['min']:.3f}, max={stats['max']:.3f}, "
        f"mean={stats['mean']:.3f}, median={stats['median']:.3f}, "
        f"std={stats['std']:.3f}, zeros={int(stats['zeros'])}, "
        f"negatives={int(stats['negatives'])}"
    )


def render_case_section(summary: CaseSummary) -> List[str]:
    lines = []
    lines.append(f"## {summary.name.upper()}")
    lines.append("")
    lines.append(f"- Rows: {summary.rows:,}")
    lines.append(f"- Columns: {summary.cols}")
    lines.append(f"- Missing values: {summary.missing_total:,} ({summary.missing_pct:.2f}%)")
    lines.append(f"- Target stats: {format_stats(summary.target_stats)}")

    if summary.categorical_cardinality:
        lines.append("- Top categorical cardinality:")
        for col, n in summary.categorical_cardinality:
            lines.append(f"  - {col}: {n}")

    if summary.numeric_correlations:
        lines.append("- Top numeric correlations (abs):")
        for col, val in summary.numeric_correlations:
            lines.append(f"  - {col}: {val:.4f}")

    if summary.leakage_flags:
        lines.append("- Leakage flags:")
        for col in summary.leakage_flags:
            lines.append(f"  - {col}")

    actions = FEATURE_ACTIONS.get(summary.name, {})
    if actions:
        lines.append("- Feature actions applied:")
        for key in ["added", "dropped", "leakage"]:
            items = actions.get(key, [])
            if items:
                joined = ", ".join(items)
                lines.append(f"  - {key}: {joined}")

    lines.append("")
    return lines


def append_results(lines: List[str]) -> List[str]:
    results_dir = OUTPUTS_DIR / "results"
    if not results_dir.exists():
        lines.append("## MODEL RESULTS")
        lines.append("")
        lines.append("- Results directory not found.")
        lines.append("")
        return lines

    lines.append("## MODEL RESULTS")
    lines.append("")

    for case in ["case1", "case2", "case3", "case4"]:
        results_path = results_dir / f"{case}_results.csv"
        lines.append(f"### {case.upper()}")
        if not results_path.exists():
            lines.append("- Results file not found.")
            lines.append("")
            continue

        results_df = pd.read_csv(results_path)
        if results_df.empty:
            lines.append("- Results file is empty.")
            lines.append("")
            continue

        if "RMSE" in results_df.columns:
            best_row = results_df.sort_values("RMSE").iloc[0]
            lines.append(f"- Best model: {best_row['model']}")
            metrics = [c for c in ["MAE", "RMSE", "R2", "MAPE", "RMSLE"] if c in results_df.columns]
            for metric in metrics:
                lines.append(f"  - {metric}: {best_row[metric]:.4f}")
        else:
            lines.append("- Metrics unavailable (missing RMSE column).")
        lines.append("")

    return lines


def main() -> None:
    summaries = []

    case1 = load_csv(CASE1_PATH)
    summaries.append(summarize_case("case1", case1, TARGET_COLUMNS["case1"]))

    case2 = load_csv(CASE2_TRAIN_PATH)
    summaries.append(summarize_case("case2", case2, TARGET_COLUMNS["case2"]))

    case3 = load_csv(CASE3_PATH, encoding="latin-1")
    summaries.append(summarize_case("case3", case3, TARGET_COLUMNS["case3"]))

    case4 = load_csv(CASE4_PATH)
    summaries.append(summarize_case("case4", case4, TARGET_COLUMNS["case4"]))

    lines: List[str] = []
    lines.append("# New Dataset Insights Report")
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

    for summary in summaries:
        lines.extend(render_case_section(summary))

    lines = append_results(lines)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
