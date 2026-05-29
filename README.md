# Context-Dependent Model Selection for Supply Chain Forecasting

Experiment code for the multi-case comparative study *"Context-Dependent Model
Selection for Supply Chain Forecasting: A Multi-Case Comparative Study of
Statistical and Machine Learning Methods"*.

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20450114.svg)](https://doi.org/10.5281/zenodo.20450114)

## Overview

This repository implements a reproducible pipeline that compares classical
statistical forecasting models with supervised machine-learning models across
four heterogeneous, publicly available supply-chain case studies. The study is
organised around three ideas that the code makes explicit:

1. **Context-dependent model selection.** No single model family dominates; the
   most suitable family depends on the structure of each task (temporal
   dominance, categorical richness, feature dimensionality).
2. **Representation asymmetry.** Classical models operate on *aggregated
   univariate series* whereas supervised models operate on *row-level
   engineered features*. The comparison is therefore framed at the level of
   modelling *pipelines*, not isolated algorithms under identical inputs.
3. **Leakage auditing.** Reported accuracy can be inflated by post-outcome
   features; the pipeline enforces a leakage-free feature set for the case most
   exposed to this risk (Case 3).

Eleven model configurations spanning ten model families are evaluated per case.

| Case | Dataset | Target | Task character |
|------|---------|--------|----------------|
| 1 | Historical Product Demand | `Order_Demand` | Time-series, temporally dominated |
| 2 | Food Demand | `num_orders` | Panel, categorical-rich |
| 3 | DataCo Supply Chain | `Days for shipping (real)` | Cross-sectional, leakage-sensitive |
| 4 | Retail Store Inventory | `Units Sold` | Panel, mixed features |

## Repository structure

```
.
├── main.py                       # Primary entry point — runs all cases and models
├── requirements.txt
├── README.md                     # This file
├── LICENSE                       # MIT
├── CITATION.cff                  # How to cite the software and the article
├── CHANGELOG.md
├── .zenodo.json                  # Metadata for Zenodo archiving / DOI
├── config/
│   ├── config.py                 # Config dataclass (paths, flags, hyperparameters)
│   └── constants.py              # Dataset paths, hyperparameters, feature specs, Case 3 leakage lists
├── data/
│   ├── data_loader.py            # Per-case dataset loading
│   ├── preprocessor.py           # Missing values, winsorisation, encoding
│   └── feature_engineer.py       # Lag/rolling/calendar features and case-specific transforms
├── models/
│   ├── base.py                   # BaseModel / TimeSeriesModel / SupervisedModel abstractions
│   ├── classical/                # Naive, Seasonal Naive, ETS, ARIMA(X), SARIMA(X), Linear Regression
│   └── ml/                       # Random Forest, XGBoost, CatBoost, Deep FFNN, Shallow ANN (PyTorch)
├── experiments/
│   ├── base_experiment.py        # Abstract run() orchestration (load → preprocess → engineer → split → train → evaluate)
│   ├── case1_experiment.py       # Canonical per-case experiments used by main.py
│   ├── case2_experiment.py
│   ├── case3_experiment.py
│   └── case4_experiment.py
├── evaluation/
│   ├── metrics.py                # MAE, RMSE, MAPE, R², RMSLE
│   └── visualizations.py         # 300-DPI figure helpers
├── utils/
│   └── helpers.py                # Seeding and timing utilities
├── paper_figures/                # Regenerates the manuscript figures from reported values
├── new_dataset/                  # Place the datasets here (see new_dataset/README.md)
└── outputs/
    └── results/
        └── all_cases_results.csv # Reported results (all cases × all models)
```

The repository root also contains a few clone-safe auxiliary utilities:
`rebuild_all_cases_results.py` recombines the per-case result files after a run,
`tune_shallow_ann.py` explores Shallow ANN hyperparameters, `new_data_insights.py`
produces exploratory data summaries, and `generate_full_documentation.py`
generates a documentation report under `outputs/reports/` when run.

## Installation

Python 3.10 is recommended.

```bash
# 0. Clone the repository
git clone https://github.com/mmahmoudai/supply-chain-forecasting-comparison.git
cd supply-chain-forecasting-comparison

# 1. Create and activate a virtual environment
python -m venv venv
# Windows:        venv\Scripts\activate
# Linux / macOS:  source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

The Deep FFNN and Shallow ANN are implemented in **PyTorch** and use the GPU
automatically when one is available. For a CUDA build, install the matching
PyTorch wheel (for example,
`pip install torch --index-url https://download.pytorch.org/whl/cu124`); the CPU
build is sufficient to reproduce the results, only slower.

## Datasets

The datasets are publicly available but are **not redistributed here**. Download
each one and place it at the exact path the code expects; the filenames and
folder layout are documented in [`new_dataset/README.md`](new_dataset/README.md)
and defined in [`config/constants.py`](config/constants.py). The consolidated
collection is also archived on the Harvard Dataverse:

- <https://dataverse.harvard.edu/dataverse/demand-forecasting-cases>

## Reproducing the results

From the repository root:

```bash
python main.py                 # all four cases, all eleven configurations
python main.py --cases 1 3     # a subset of cases
python main.py --no-neural     # skip the PyTorch FFNN / Shallow ANN models
python main.py --no-classical  # machine-learning models only
python main.py --tune          # enable hyperparameter search (slower)
python main.py --help          # all options
```

`main.py` runs each case experiment, evaluates every model, and writes the
combined metrics to `outputs/results/all_cases_results.csv` together with
`summary.json` and a LaTeX table (`results_table.tex`). The
`outputs/results/all_cases_results.csv` committed to this repository contains
the values reported in the manuscript.

**Reproducibility note.** All experiments set a fixed random seed
(`RANDOM_SEED = 42`) and an 80/20 split (`TRAIN_RATIO = 0.8`); Case 2 uses the
week-based splits defined in `config/constants.py`. Results should match the
reported values closely, though small numerical differences can occur across
hardware, BLAS, and library versions (particularly for GPU-trained neural
models). The committed results file is the reference.

**Case 2 note.** Case 2 reads the consolidated training file
`case_2_food_Demand_train_v3.csv` (the prepared version of the public food-demand
dataset used in the study); see [`new_dataset/README.md`](new_dataset/README.md).

## Models and configurations

Models are evaluated in a fixed order across all cases:

> Naive → Seasonal Naive → ETS → ARIMA(X) → SARIMA(X) → Linear Regression →
> Random Forest → XGBoost → CatBoost → Shallow ANN → Deep FFNN

The two neural configurations (Shallow ANN and Deep FFNN) belong to a single
feed-forward family, giving **eleven configurations across ten families**. Key
default hyperparameters (from `config/constants.py`):

| Model | Key defaults |
|-------|--------------|
| XGBoost | 200 trees, learning rate 0.05, max depth 5 |
| CatBoost | 200 iterations, learning rate 0.05, depth 6 |
| Random Forest | 200 trees, max depth 20 |
| Deep FFNN | hidden layers [128, 64, 32], learning rate 0.001, 200 epochs, batch 1024 |
| Shallow ANN | 96 hidden units, learning rate 0.0005, 150 epochs |

CatBoost is supplied the **raw categorical columns** through its native
categorical handling in the categorical-rich cases (Cases 2–4), whereas the
other supervised models use one-hot / target-encoded features. This deliberate
asymmetry is part of the pipeline-level framing discussed in the paper.

## Evaluation metrics

The code computes **MAE, RMSE, MAPE, R²**, and (where applicable) **RMSLE**
(`evaluation/metrics.py`). The manuscript additionally reports **rMAE**, a
naive-benchmark-relative error defined as

```
rMAE = MAE(model) / MAE(naive benchmark)
```

evaluated on the same set, so that the Naive model equals 1.000 by construction.
rMAE is derived from the MAE values in `all_cases_results.csv`; it is not stored
as a separate column.

## Leakage auditing

Leakage is audited in every case. Case 3 (shipping-duration prediction) enforces
a strict leakage-free feature set: outcome and post-outcome variables — delivery
status, late-delivery risk, shipping dates, profit ratios — and identifiers are
excluded by design (see `CASE3_APPROVED_COLS` and `CASE3_EXCLUDED_COLS` in
`config/constants.py`). In Case 4, admitting the pre-computed `Demand Forecast`
feature — effectively a copy of the target — inflates the best model's R² from
about 0.94 to 0.34 (ΔR² ≈ 0.596, roughly 60 percentage points), which motivates
leakage auditing as a standard evaluation step.

## Citation

If you use this software, please cite both the article and the software archive.

```bibtex
@article{helmy2026context,
  title   = {Context-Dependent Model Selection for Supply Chain Forecasting:
             A Multi-Case Comparative Study of Statistical and Machine Learning Methods},
  author  = {Helmy, AbdelMoniem and Mahmoud, Muhammad},
  year    = {2026},
  note    = {Manuscript under review}
}

@software{helmy2026code,
  title     = {Context-Dependent Model Selection for Supply Chain Forecasting: Experiment Code},
  author    = {Helmy, AbdelMoniem and Mahmoud, Muhammad},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20450114}
}
```

A machine-readable citation is provided in [`CITATION.cff`](CITATION.cff).

## Authors

- **AbdelMoniem Helmy** — Faculty of Graduate Studies for Statistical Research,
  Cairo University, Cairo, Egypt (corresponding author).
- **Muhammad Mahmoud** — Faculty of Computers and Artificial Intelligence,
  Matrouh University, Matrouh, Egypt.

## License

Released under the [MIT License](LICENSE). The datasets remain subject to their
original providers' terms; see [`new_dataset/README.md`](new_dataset/README.md).
