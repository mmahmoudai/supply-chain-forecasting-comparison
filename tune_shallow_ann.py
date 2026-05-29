"""Tune shallow ANN hyperparameters for Case 1 and Case 2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from config.constants import (
    OUTPUTS_DIR,
    TARGET_COLUMNS,
    TRAIN_RATIO,
    CASE2_TRAIN_WEEKS,
    CASE2_VAL_WEEKS
)
from data.data_loader import DataLoader
from data.preprocessor import Preprocessor
from data.feature_engineer import FeatureEngineer
from models.ml import ShallowANNModel


REPORT_DIR = OUTPUTS_DIR / "reports"
CSV_PATH = REPORT_DIR / "shallow_ann_tuning.csv"
MD_PATH = REPORT_DIR / "shallow_ann_tuning.md"
SAMPLE_FRAC = 0.3


def _sample_time_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sample the earliest fraction of a time-ordered dataframe."""
    n = max(1, int(len(df) * SAMPLE_FRAC))
    return df.iloc[:n].copy()


@dataclass
class TuningResult:
    params: Dict
    rmse_case1: float
    rmse_case2: float
    rank_sum: int


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def prepare_case1() -> Dict[str, np.ndarray]:
    loader = DataLoader()
    preprocessor = Preprocessor()
    fe = FeatureEngineer()

    df = loader.load_case1()
    df = preprocessor.preprocess_case1(df)
    df = fe.engineer_case1_features(df, aggregate=True)

    target = TARGET_COLUMNS['case1']
    date_col = 'Date'
    df = df.sort_values(date_col)

    split_idx = int(len(df) * TRAIN_RATIO)
    train = df.iloc[:split_idx].copy()

    # Use last 20% of train as validation (time-based)
    val_idx = int(len(train) * 0.8)
    train_part = train.iloc[:val_idx]
    val_part = train.iloc[val_idx:]

    # Sample for faster tuning
    train_part = _sample_time_df(train_part)
    val_part = _sample_time_df(val_part)

    feature_cols = fe.get_feature_names('case1')

    return {
        'X_train': train_part[feature_cols],
        'y_train': train_part[target].values,
        'X_val': val_part[feature_cols],
        'y_val': val_part[target].values
    }


def prepare_case2() -> Dict[str, np.ndarray]:
    loader = DataLoader()
    preprocessor = Preprocessor()
    fe = FeatureEngineer()

    train_df, _ = loader.load_case2(include_test=False)
    train_df = preprocessor.preprocess_case2(train_df)
    train_df, _ = fe.engineer_case2_features(train_df, None)

    target = TARGET_COLUMNS['case2']

    train = train_df[(train_df['week'] >= CASE2_TRAIN_WEEKS[0]) &
                     (train_df['week'] <= CASE2_TRAIN_WEEKS[1])].copy()
    val = train_df[(train_df['week'] >= CASE2_VAL_WEEKS[0]) &
                   (train_df['week'] <= CASE2_VAL_WEEKS[1])].copy()
    train = train.sort_values('week')
    val = val.sort_values('week')

    exclude_cols = [target, 'log_num_orders', 'week', 'center_id', 'meal_id']
    feature_cols = [c for c in train.columns if c not in exclude_cols]
    feature_cols = [c for c in feature_cols
                    if train[c].dtype in ['float64', 'int64', 'float32', 'int32', 'uint8']]

    train = train.dropna(subset=feature_cols + [target, 'log_num_orders'])
    val = val.dropna(subset=feature_cols + [target, 'log_num_orders'])

    train = _sample_time_df(train)
    val = _sample_time_df(val)

    return {
        'X_train': train[feature_cols],
        'y_train': train['log_num_orders'].values,
        'X_val': val[feature_cols],
        'y_val': val['log_num_orders'].values,
        'y_val_original': val[target].values
    }


def main() -> None:
    candidates = [
        {
            'hidden_units': 64,
            'learning_rate': 1e-3,
            'dropout_rate': 0.1,
            'epochs': 150,
            'patience': 15
        },
        {
            'hidden_units': 64,
            'learning_rate': 5e-4,
            'dropout_rate': 0.1,
            'epochs': 150,
            'patience': 15
        },
        {
            'hidden_units': 96,
            'learning_rate': 5e-4,
            'dropout_rate': 0.1,
            'epochs': 150,
            'patience': 15
        },
        {
            'hidden_units': 128,
            'learning_rate': 5e-4,
            'dropout_rate': 0.1,
            'epochs': 200,
            'patience': 20
        },
        {
            'hidden_units': 128,
            'learning_rate': 5e-4,
            'dropout_rate': 0.2,
            'epochs': 200,
            'patience': 20
        },
        {
            'hidden_units': 128,
            'learning_rate': 3e-4,
            'dropout_rate': 0.2,
            'epochs': 200,
            'patience': 20
        }
    ]

    case1 = prepare_case1()
    case2 = prepare_case2()

    results: List[TuningResult] = []

    for params in candidates:
        model = ShallowANNModel(**params)
        model.fit(
            case1['X_train'], case1['y_train'],
            X_val=case1['X_val'], y_val=case1['y_val'],
            verbose=0
        )
        preds1 = model.predict(case1['X_val'])
        rmse1 = rmse(case1['y_val'], preds1)

        model2 = ShallowANNModel(**params)
        model2.fit(
            case2['X_train'], case2['y_train'],
            X_val=case2['X_val'], y_val=case2['y_val'],
            verbose=0
        )
        preds2_log = model2.predict(case2['X_val'])
        preds2 = np.expm1(preds2_log)
        preds2 = np.maximum(preds2, 0)
        rmse2 = rmse(case2['y_val_original'], preds2)

        results.append(TuningResult(params=params, rmse_case1=rmse1, rmse_case2=rmse2, rank_sum=0))

    # Rank by RMSE per case
    rmse1_values = [r.rmse_case1 for r in results]
    rmse2_values = [r.rmse_case2 for r in results]

    rank1 = {v: i + 1 for i, v in enumerate(sorted(rmse1_values))}
    rank2 = {v: i + 1 for i, v in enumerate(sorted(rmse2_values))}

    for result in results:
        result.rank_sum = rank1[result.rmse_case1] + rank2[result.rmse_case2]

    results_sorted = sorted(results, key=lambda r: r.rank_sum)
    best = results_sorted[0]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results_sorted:
        row = {
            **result.params,
            'rmse_case1': result.rmse_case1,
            'rmse_case2': result.rmse_case2,
            'rank_sum': result.rank_sum
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(CSV_PATH, index=False)

    lines = [
        "# Shallow ANN Tuning Report",
        "",
        f"Sample fraction used: {SAMPLE_FRAC}",
        f"Best params (rank sum): {best.params}",
        f"Case 1 RMSE (val): {best.rmse_case1:.4f}",
        f"Case 2 RMSE (val): {best.rmse_case2:.4f}",
        "",
        "## All trials",
        "",
        df.to_string(index=False)
    ]
    MD_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
