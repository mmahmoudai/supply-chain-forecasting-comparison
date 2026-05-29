# Manuscript figures

`generate_figures.py` reproduces the figures used in the manuscript
(cross-case performance heatmap, average-rank chart, R-squared comparison, and
the leakage-impact panel).

This script is a **presentation layer, not part of the modelling pipeline**: it
renders figures from the summary values reported in the paper (which are
embedded in the script and correspond to `outputs/results/all_cases_results.csv`).
Running the experiments via [`main.py`](../main.py) is what regenerates those
underlying numbers; this script only redraws them in publication form.

## Usage

```bash
cd paper_figures
python generate_figures.py
```

Output figures are written next to the script (PDF/PNG).
