# Datasets

The four datasets used in this study are **publicly available but are not
redistributed in this repository** (they are subject to their original
providers' licences, and one is large). To reproduce the experiments, obtain
each dataset and place it at the exact path the code expects, as listed below.

The consolidated dataset collection is also archived on the Harvard Dataverse,
and the original sources are summarised in the manuscript's *Data Availability*
statement.

- Consolidated collection: <https://dataverse.harvard.edu/dataverse/demand-forecasting-cases>

## Expected files and paths

Paths are relative to the repository root. The exact filenames are defined in
[`config/constants.py`](../config/constants.py) (`CASE1_PATH`, `CASE2_TRAIN_PATH`,
`CASE3_PATH`, `CASE4_PATH`).

| Case | Expected path | Target variable | Original source |
|------|---------------|-----------------|-----------------|
| 1 — Historical Product Demand | `new_dataset/case 1/case 1-Historical Product Demand.csv` | `Order_Demand` | Kaggle (historical product demand) |
| 2 — Food Demand | `new_dataset/case 2/new_22032026/case_2_food_Demand_train_v3.csv` | `num_orders` | Kaggle (food demand forecasting) — consolidated training file used in the study |
| 3 — DataCo Supply Chain | `new_dataset/case 3/DataCoSupplyChainDataset.csv` | `Days for shipping (real)` | Mendeley Data (DataCo smart supply chain) |
| 4 — Retail Store Inventory | `new_dataset/case 4/retail_store_inventory.csv` | `Units Sold` | Kaggle (retail store inventory) |

A `.gitkeep` file marks each expected folder so the directory layout is
preserved; drop the corresponding CSV into the folder before running.

## Notes

- **Case 2** expects a consolidated training file (`case_2_food_Demand_train_v3.csv`),
  which is the prepared version of the public food-demand dataset used in the
  study. The same loader path is used for the held-out predictions, as there is
  no separate test file.
- **Case 3** uses a strict leakage-free feature set. Outcome and post-outcome
  variables (e.g. delivery status, late-delivery risk, shipping dates, profit
  ratios) are excluded by design; see `CASE3_APPROVED_COLS` and
  `CASE3_EXCLUDED_COLS` in [`config/constants.py`](../config/constants.py).
