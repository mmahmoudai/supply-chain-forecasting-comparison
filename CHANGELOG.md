# Changelog

All notable changes to this project are documented in this file. The format is
loosely based on [Keep a Changelog](https://keepachangelog.com/), and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- Added the Zenodo archive DOI (README badge, `@software` citation entry, and the `CITATION.cff` `doi` field).
- Removed the target-venue reference from the project metadata while the associated paper is under review.

## [1.0.0] — 2026-05-29

Initial public release accompanying the manuscript *"Context-Dependent Model
Selection for Supply Chain Forecasting: A Multi-Case Comparative Study of
Statistical and Machine Learning Methods"*.

### Added
- End-to-end experimental pipeline (`main.py`) covering four supply-chain case
  studies and eleven model configurations across ten model families.
- Classical models: Naive, Seasonal Naive, ETS, ARIMA(X), SARIMA(X), Linear
  Regression. Machine-learning models: Random Forest, XGBoost, CatBoost, Deep
  FFNN, and Shallow ANN (the two neural configurations implemented in PyTorch).
- Case-specific preprocessing, feature engineering, and the leakage-auditing
  protocol applied to Case 3 (DataCo shipping-duration prediction).
- Reported results in `outputs/results/all_cases_results.csv`.
- Manuscript figure-regeneration script under `paper_figures/`.
