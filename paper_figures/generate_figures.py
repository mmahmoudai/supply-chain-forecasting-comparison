"""
Regenerate the manuscript figures from the reported results.

All per-case metric values are read from the single source of truth,
``outputs/results/all_cases_results.csv`` (relative to the repository root),
rather than being hard-coded:

  * Figure 2 (rMAE heatmap)  -- rMAE = model MAE / Naive MAE, per case.
  * Figure 4 (R^2 bars)      -- R^2 read directly from the results file.
  * Figure 5 (leakage, Case 4) -- the post-removal R^2 is the best supervised
    R^2 in Case 4 read from the file.

Two values are *not* recomputed here, and are documented inline where used:

  * Figure 3 (average ranks) mirrors the average model ranks reported in
    Table 6 of the manuscript. Those ranks are a deployment-oriented aggregate
    from the experiment pipeline; because the Case 4 supervised models are
    essentially tied, the ranking is not a single-metric sort of the results
    file. The per-case ranks from Table 6 are encoded below and the averages
    are computed from them.
  * Figure 5's pre-removal R^2 comes from a separate run in which the leaking
    ``Demand Forecast`` feature was included; that run is not part of the
    leakage-free results file.

Usage:
    cd paper_figures
    python generate_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Publication settings
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ============================================================
# Load the reported results (single source of truth)
# ============================================================
RESULTS_CSV = (Path(__file__).resolve().parent.parent
               / 'outputs' / 'results' / 'all_cases_results.csv')
if not RESULTS_CSV.exists():
    raise SystemExit(
        f"Results file not found: {RESULTS_CSV}\n"
        "Run 'python main.py' from the repository root to generate it.")

results = pd.read_csv(RESULTS_CSV)
CASES = ['case1', 'case2', 'case3', 'case4']

# Display label -> model name as stored in the results file.
CSV_NAME = {
    'Naive': 'Naive', 'S.Naive': 'SeasonalNaive', 'ETS': 'ETS',
    'ARIMAX': 'ARIMAX', 'SARIMAX': 'SARIMAX', 'LR': 'LinearRegression',
    'RF': 'RandomForest', 'XGBoost': 'XGBoost', 'CatBoost': 'CatBoost',
    'S.FFNN': 'Shallow NN', 'D.FFNN': 'Deep NN',
}


def value(case, label, column):
    """Return ``column`` for a display ``label`` in a given ``case``."""
    row = results[(results['case'] == case) & (results['model'] == CSV_NAME[label])]
    if row.empty:
        raise KeyError(f"No row for model '{CSV_NAME[label]}' in {case}")
    return float(row[column].iloc[0])


# ============================================================
# FIGURE 2: Cross-case rMAE heatmap
# ============================================================
models = ['Naive', 'S.Naive', 'ETS', 'ARIMAX', 'SARIMAX',
          'LR', 'RF', 'XGBoost', 'CatBoost', 'S.FFNN', 'D.FFNN']
cases = ['Case 1:\nProduct', 'Case 2:\nFood', 'Case 3:\nShipping', 'Case 4:\nRetail']

# rMAE = model MAE / Naive MAE (per case); rounded to the reported precision.
rmae_data = np.zeros((len(models), len(CASES)))
for j, case in enumerate(CASES):
    naive_mae = value(case, 'Naive', 'MAE')
    for i, label in enumerate(models):
        rmae_data[i, j] = round(value(case, label, 'MAE') / naive_mae, 3)

fig, ax = plt.subplots(figsize=(8, 6))
# Use log scale for better visualization
rmae_log = np.log10(rmae_data + 0.0001)
im = ax.imshow(rmae_log, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(4))
ax.set_xticklabels(cases)
ax.set_yticks(range(11))
ax.set_yticklabels(models)
# Annotate cells with actual rMAE values
for i in range(11):
    for j in range(4):
        val = rmae_data[i, j]
        txt = f'{val:.3f}' if val >= 0.01 else f'{val:.4f}'
        color = 'white' if rmae_log[i, j] > 0 else 'black'
        ax.text(j, i, txt, ha='center', va='center', fontsize=7.5, color=color)
ax.set_title('Cross-Case rMAE Performance Heatmap (log-scale colouring)')
# Add horizontal line separating stat from supervised
ax.axhline(y=4.5, color='black', linewidth=1.5, linestyle='--')
ax.text(3.6, 2, 'Statistical', fontsize=8, ha='right', style='italic', color='gray')
ax.text(3.6, 7.5, 'Supervised', fontsize=8, ha='right', style='italic', color='gray')
plt.colorbar(im, ax=ax, label='log$_{10}$(rMAE)', shrink=0.8)
plt.tight_layout()
plt.savefig('fig2_mase_heatmap.pdf')
plt.savefig('fig2_mase_heatmap.png')
plt.close()
print('Figure 2: rMAE heatmap saved.')

# ============================================================
# FIGURE 3: Average model rank bar chart
# ============================================================
# Per-case ranks as reported in Table 6 (tab:avg_rank) of the manuscript. These
# are a deployment-oriented aggregate from the experiment pipeline; the Case 4
# supervised models are essentially tied, so the ranking is not reproducible as a
# single-metric sort of the results file. The averages are computed from them, and
# the figure caption states that it uses the values from Table 6.
per_case_ranks = {
    'CatBoost': [1, 1, 8, 2],
    'RF':       [2, 2, 6, 5],
    'LR':       [3, 6, 11, 1],
    'XGBoost':  [5, 5, 7, 4],
    'S.FFNN':   [4, 4, 10, 6],
    'ARIMAX':   [7, 7, 3, 8],
    'D.FFNN':   [11, 3, 9, 3],
    'ETS':      [8, 10, 2, 7],
    'SARIMAX':  [6, 11, 1, 9],
    'S.Naive':  [9, 9, 5, 10],
    'Naive':    [10, 8, 4, 11],
}
avg_rank = {m: sum(r) / len(r) for m, r in per_case_ranks.items()}
# Sort ascending by average rank; stable sort preserves Table 6 order on ties.
rank_models = sorted(per_case_ranks, key=lambda m: avg_rank[m])
avg_ranks = [avg_rank[m] for m in rank_models]

# Colour by model family.
family_color = {
    'CatBoost': '#2ecc71', 'RF': '#27ae60', 'XGBoost': '#1e8449',  # tree-based
    'LR': '#3498db',                                               # other supervised
    'S.FFNN': '#e67e22', 'D.FFNN': '#e67e22',                      # neural network
    'ARIMAX': '#95a5a6', 'ETS': '#95a5a6', 'SARIMAX': '#95a5a6',   # statistical
    'S.Naive': '#bdc3c7', 'Naive': '#bdc3c7',                      # naive baselines
}
colors_list = [family_color[m] for m in rank_models]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.barh(range(len(rank_models)), avg_ranks, color=colors_list, edgecolor='white', height=0.7)
ax.set_yticks(range(len(rank_models)))
ax.set_yticklabels(rank_models)
ax.set_xlabel('Average Rank (lower is better)')
ax.set_title('Cross-Case Average Model Rank')
ax.invert_yaxis()
ax.set_xlim(0, 10)
for i, v in enumerate(avg_ranks):
    ax.text(v + 0.1, i, f'{v:.2f}', va='center', fontsize=9)
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='Tree-based ML'),
                   Patch(facecolor='#3498db', label='Other supervised'),
                   Patch(facecolor='#e67e22', label='Neural network'),
                   Patch(facecolor='#95a5a6', label='Statistical')]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('fig3_avg_rank.pdf')
plt.savefig('fig3_avg_rank.png')
plt.close()
print('Figure 3: Average rank chart saved.')

# ============================================================
# FIGURE 4: R^2 comparison across cases
# ============================================================
supervised_models = ['LR', 'RF', 'XGBoost', 'CatBoost', 'S.FFNN', 'D.FFNN']
case_title = {'case1': 'Case 1', 'case2': 'Case 2', 'case3': 'Case 3', 'case4': 'Case 4'}
r2_data = {case_title[c]: [value(c, m, 'R2') for m in supervised_models] for c in CASES}

fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
for idx, (case_name, r2_vals) in enumerate(r2_data.items()):
    ax = axes[idx // 2, idx % 2]
    r2_clipped = [max(v, -0.5) for v in r2_vals]
    colors_r2 = ['#3498db' if v > 0 else '#e74c3c' for v in r2_vals]
    bars = ax.bar(range(len(supervised_models)), r2_clipped, color=colors_r2, edgecolor='white')
    # Hatch bars whose true R2 is below the -0.5 display floor (off-scale / clipped)
    for b, v in zip(bars, r2_vals):
        if v < -0.5:
            b.set_hatch('///')
    ax.set_title(case_name, fontweight='bold')
    ax.set_ylabel('$R^2$')
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')
    ax.set_xticks(range(len(supervised_models)))
    ax.set_xticklabels(supervised_models, rotation=30, ha='right', fontsize=8)
    ax.set_ylim(min(r2_clipped) - 0.14, max(r2_clipped) + 0.14)
    for i, v in enumerate(r2_vals):
        txt = f'{v:.3f}' if abs(v) < 1 else f'{v:.2f}'
        if v < -0.5:
            ax.text(i, -0.48, txt, ha='center', va='top', fontsize=7,
                    color='#7b241c', fontweight='bold')
        else:
            ax.text(i, max(v, -0.5) + 0.02, txt, ha='center', va='bottom', fontsize=7.5)
fig.suptitle('$R^2$ Performance of Supervised Models Across Cases', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('fig4_r2_comparison.pdf')
plt.savefig('fig4_r2_comparison.png')
plt.close()
print('Figure 4: R^2 comparison saved.')

# ============================================================
# FIGURE 5: Leakage detection impact (Case 4)
# ============================================================
# Post-removal R^2: best supervised R^2 in the leakage-free Case 4 results.
after_removal = round(max(value('case4', m, 'R2') for m in supervised_models), 3)
# Pre-removal R^2: from a separate run with the leaking 'Demand Forecast' feature
# included. That run is not in the leakage-free results file; this is the value
# reported in the manuscript.
before_removal = 0.94
delta_r2 = round(before_removal - after_removal, 3)

fig, ax = plt.subplots(figsize=(6, 4))
categories = ['Before Leakage\nRemoval', 'After Leakage\nRemoval']
r2_values = [before_removal, after_removal]
bar_colors = ['#e74c3c', '#2ecc71']
bars = ax.bar(categories, r2_values, color=bar_colors, width=0.5, edgecolor='white')
ax.set_ylabel('$R^2$')
ax.set_title('Case 4: Impact of Data Leakage on $R^2$ (Best Model)')
ax.set_ylim(0, 1.1)
for bar, val in zip(bars, r2_values):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f'{val:.3f}',
            ha='center', fontsize=12, fontweight='bold')
# Arrow showing the drop
ax.annotate('', xy=(1, after_removal + 0.04), xytext=(0, before_removal - 0.04),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))
ax.text(0.5, 0.65, f'$\\Delta R^2 = {delta_r2:.3f}$\n($\\approx${round(delta_r2 * 100)} pp inflation)',
        ha='center', fontsize=10, style='italic',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig('fig5_leakage_impact.pdf')
plt.savefig('fig5_leakage_impact.png')
plt.close()
print('Figure 5: Leakage impact saved.')

print('\nAll figures generated successfully from', RESULTS_CSV.name)
