"""Visualization functions for EDA and model evaluation."""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter threading issues

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.constants import FIGURE_DPI, FIGURE_STYLE, DEFAULT_FIGSIZE, FIGURES_DIR


class Visualizer:
    """Publication-quality visualizations for thesis chapters."""
    
    def __init__(self, 
                 output_dir: Path = FIGURES_DIR,
                 dpi: int = FIGURE_DPI,
                 style: str = FIGURE_STYLE):
        """
        Initialize the Visualizer.
        
        Args:
            output_dir: Directory to save figures
            dpi: Figure DPI for saving
            style: Matplotlib style to use
        """
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.style = style
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create output directories
        (self.output_dir / 'chapter6').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'chapter7').mkdir(parents=True, exist_ok=True)
        
        # Set style
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8-whitegrid')
        
        # Set default parameters - INCREASED FONT SIZES per PDF comment
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['savefig.dpi'] = dpi
        plt.rcParams['font.size'] = 14          # Increased from 12
        plt.rcParams['axes.labelsize'] = 14     # Increased from 12
        plt.rcParams['axes.titlesize'] = 16     # Increased from 14
        plt.rcParams['legend.fontsize'] = 12    # Increased from 10
        plt.rcParams['xtick.labelsize'] = 12    # Added
        plt.rcParams['ytick.labelsize'] = 12    # Added
    
    def _save_figure(self, fig: plt.Figure, filename: str, 
                     chapter: str = 'chapter6'):
        """Save figure to the appropriate directory."""
        filepath = self.output_dir / chapter / f"{filename}.png"
        fig.savefig(filepath, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        self.logger.info(f"Saved figure: {filepath}")
    
    def plot_time_series(self, data: pd.DataFrame,
                         date_col: str,
                         value_col: str,
                         title: str,
                         filename: str,
                         chapter: str = 'chapter6',
                         figsize: Tuple = DEFAULT_FIGSIZE):
        """
        Plot a time series line chart.
        
        Args:
            data: DataFrame with time series data
            date_col: Name of date column
            value_col: Name of value column
            title: Figure title
            filename: Output filename (without extension)
            chapter: 'chapter6' or 'chapter7'
            figsize: Figure size tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(data[date_col], data[value_col], linewidth=1.5, color='#2E86AB', label=value_col)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(value_col, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, filename, chapter)
    
    def plot_distribution(self, data: Union[pd.Series, np.ndarray],
                          title: str,
                          filename: str,
                          xlabel: str = 'Value',
                          chapter: str = 'chapter6',
                          figsize: Tuple = DEFAULT_FIGSIZE):
        """
        Plot histogram with KDE overlay.
        
        Args:
            data: Series or array of values
            title: Figure title
            filename: Output filename
            xlabel: X-axis label
            chapter: Output chapter directory
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.histplot(data, kde=True, ax=ax, color='#2E86AB', edgecolor='white',
                    label='Histogram', line_kws={'label': 'KDE', 'linewidth': 2})
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, filename, chapter)
    
    def plot_actual_vs_predicted(self, y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  model_name: str,
                                  title: str,
                                  filename: str,
                                  chapter: str = 'chapter7',
                                  figsize: Tuple = DEFAULT_FIGSIZE):
        """
        Plot actual vs predicted scatter plot.
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            model_name: Name of the model
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=20, color='#2E86AB', label=model_name)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(title)
        ax.legend()
        
        plt.tight_layout()
        self._save_figure(fig, filename, chapter)
    
    def plot_residuals(self, y_true: np.ndarray,
                       y_pred: np.ndarray,
                       title: str,
                       filename: str,
                       chapter: str = 'chapter7',
                       figsize: Tuple = (12, 5)):
        """
        Plot residual analysis (residuals vs fitted and histogram).
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Figure size
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Residuals vs Fitted
        axes[0].scatter(y_pred, residuals, alpha=0.5, s=20, color='#2E86AB')
        axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Predicted Values')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs Fitted')
        
        # Residual histogram
        sns.histplot(residuals, kde=True, ax=axes[1], color='#2E86AB', edgecolor='white')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Residual Distribution')
        
        fig.suptitle(title, fontsize=14, y=1.02)
        plt.tight_layout()
        self._save_figure(fig, filename, chapter)
    
    def plot_model_comparison_line(self, data: pd.DataFrame,
                                    date_col: str,
                                    actual_col: str,
                                    predictions: Dict[str, np.ndarray],
                                    title: str,
                                    filename: str,
                                    chapter: str = 'chapter7',
                                    figsize: Tuple = (12, 6)):
        """
        Plot actual vs multiple model predictions as line chart.
        
        Args:
            data: DataFrame with actual values
            date_col: Date column name
            actual_col: Actual values column name
            predictions: Dict of {model_name: predictions_array}
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.tab10.colors
        
        # Plot actual
        ax.plot(data[date_col], data[actual_col], 
                label='Actual', linewidth=2, color='black')
        
        # Plot predictions
        for i, (model_name, preds) in enumerate(predictions.items()):
            ax.plot(data[date_col].iloc[-len(preds):], preds,
                   label=model_name, linewidth=1.5, linestyle='--',
                   color=colors[i % len(colors)])
        
        ax.set_xlabel('Date')
        ax.set_ylabel(actual_col)
        ax.set_title(title)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self._save_figure(fig, filename, chapter)
    
    def plot_error_boxplot(self, errors: Dict[str, np.ndarray],
                           title: str,
                           filename: str,
                           chapter: str = 'chapter7',
                           figsize: Tuple = DEFAULT_FIGSIZE):
        """
        Plot boxplot of absolute errors for model comparison.
        
        Args:
            errors: Dict of {model_name: absolute_errors_array}
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        error_df = pd.DataFrame(errors)
        error_df_melted = error_df.melt(var_name='Model', value_name='Absolute Error')
        
        sns.boxplot(x='Model', y='Absolute Error', data=error_df_melted, ax=ax,
                   palette='Set2')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        self._save_figure(fig, filename, chapter)

    def plot_error_distribution_kde(self, errors: Dict[str, np.ndarray],
                                     title: str,
                                     filename: str,
                                     chapter: str = 'chapter7',
                                     figsize: Tuple = DEFAULT_FIGSIZE):
        """
        Plot error distributions using histograms with KDE overlay (Figure 7.2).

        Per PDF specification: Use histograms or kernel density estimates to
        compare error distributions between classical and intelligent models.

        Args:
            errors: Dict of {model_name: forecast_errors_array (actual - predicted)}
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = ['#2E86AB', '#E94F37', '#44BBA4', '#393E41', '#F39C12']

        for i, (model_name, err) in enumerate(errors.items()):
            color = colors[i % len(colors)]
            sns.histplot(err, kde=True, ax=ax, color=color, alpha=0.4,
                        label=model_name, edgecolor='white', stat='density')

        ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.set_xlabel('Forecast Error (Actual - Predicted)')
        ax.set_ylabel('Density')
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()
        self._save_figure(fig, filename, chapter)

    def plot_feature_importance(self, importance_df: pd.DataFrame,
                                 title: str,
                                 filename: str,
                                 top_n: int = 15,
                                 chapter: str = 'chapter7',
                                 figsize: Tuple = (10, 8)):
        """
        Plot horizontal bar chart of feature importance.
        
        Args:
            importance_df: DataFrame with 'feature' and 'importance' columns
            title: Figure title
            filename: Output filename
            top_n: Number of top features to show
            chapter: Output chapter directory
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get top N features
        df = importance_df.nlargest(top_n, 'importance')
        
        # Plot horizontal bar chart
        bars = ax.barh(df['feature'], df['importance'], color='#2E86AB')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title(title)
        ax.invert_yaxis()  # Highest importance at top
        
        plt.tight_layout()
        self._save_figure(fig, filename, chapter)
    
    def plot_metrics_heatmap(self, metrics_df: pd.DataFrame,
                              title: str,
                              filename: str,
                              chapter: str = 'chapter7',
                              figsize: Tuple = (12, 8),
                              metric_name: Optional[str] = None):
        """
        Plot heatmap of model performance metrics.
        
        Args:
            metrics_df: DataFrame with models as rows and metrics as columns
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Figure size
            metric_name: Name of the metric when columns are cases (e.g., RMSE, R2)
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize metrics for better visualization
        df_normalized = metrics_df.copy()
        if metric_name:
            metric_upper = metric_name.upper()
            if metric_upper == 'R2':
                df_normalized = df_normalized / df_normalized.max()
            else:
                # Invert for metrics where lower is better (e.g., RMSE/MAE/MAPE)
                df_normalized = 1 - (df_normalized / df_normalized.max())
        else:
            for col in df_normalized.columns:
                if col != 'R2':
                    # Invert for metrics where lower is better
                    df_normalized[col] = 1 - (df_normalized[col] / df_normalized[col].max())
                else:
                    df_normalized[col] = df_normalized[col] / df_normalized[col].max()
        
        sns.heatmap(df_normalized, annot=metrics_df.round(4), fmt='', 
                   cmap='RdYlGn', ax=ax, cbar_kws={'label': 'Normalized Score'})
        ax.set_title(title)
        
        plt.tight_layout()
        self._save_figure(fig, filename, chapter)
    
    def plot_learning_curve(self, train_sizes: List[float],
                            train_scores: List[float],
                            val_scores: List[float],
                            title: str,
                            filename: str,
                            chapter: str = 'chapter7',
                            figsize: Tuple = DEFAULT_FIGSIZE):
        """
        Plot learning curve showing model performance vs training size.
        
        Args:
            train_sizes: List of training set sizes (as fractions)
            train_scores: Training scores for each size
            val_scores: Validation scores for each size
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(train_sizes, train_scores, 'o-', label='Training Score', 
                color='#2E86AB', linewidth=2, markersize=8)
        ax.plot(train_sizes, val_scores, 'o-', label='Validation Score',
                color='#E94F37', linewidth=2, markersize=8)
        
        ax.set_xlabel('Training Set Size (fraction)')
        ax.set_ylabel('R² Score')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, filename, chapter)
    
    def plot_seasonal_decomposition(self, data: pd.Series,
                                     title: str,
                                     filename: str,
                                     period: int = 12,
                                     chapter: str = 'chapter7',
                                     figsize: Tuple = (12, 10)):
        """
        Plot seasonal decomposition of time series.
        
        Args:
            data: Time series data
            title: Figure title
            filename: Output filename
            period: Seasonal period
            chapter: Output chapter directory
            figsize: Figure size
        """
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        try:
            decomposition = seasonal_decompose(data, model='additive', period=period)
            
            fig, axes = plt.subplots(4, 1, figsize=figsize)
            
            decomposition.observed.plot(ax=axes[0], color='#2E86AB')
            axes[0].set_ylabel('Observed')
            axes[0].set_title(title)
            
            decomposition.trend.plot(ax=axes[1], color='#E94F37')
            axes[1].set_ylabel('Trend')
            
            decomposition.seasonal.plot(ax=axes[2], color='#44BBA4')
            axes[2].set_ylabel('Seasonal')
            
            decomposition.resid.plot(ax=axes[3], color='#393E41')
            axes[3].set_ylabel('Residual')
            
            plt.tight_layout()
            self._save_figure(fig, filename, chapter)
            
        except Exception as e:
            self.logger.error(f"Could not perform seasonal decomposition: {e}")
    
    def plot_correlation_matrix(self, data: pd.DataFrame,
                                 title: str,
                                 filename: str,
                                 chapter: str = 'chapter6',
                                 figsize: Tuple = (12, 10)):
        """
        Plot correlation matrix heatmap.
        
        Args:
            data: DataFrame with numeric columns
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Select only numeric columns
        numeric_df = data.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0, ax=ax,
                   square=True, linewidths=0.5)
        ax.set_title(title)

        plt.tight_layout()
        self._save_figure(fig, filename, chapter)

    def plot_lagged_scatter(self, data: pd.DataFrame,
                            lag_col: str,
                            target_col: str,
                            title: str,
                            filename: str,
                            chapter: str = 'chapter6',
                            add_regression: bool = True,
                            figsize: Tuple = DEFAULT_FIGSIZE):
        """
        Plot lagged demand vs future demand relationship (Figure 6.7).

        Args:
            data: DataFrame with lag and target columns
            lag_col: Name of lagged feature column
            target_col: Name of target column
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            add_regression: Whether to add regression line
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot
        ax.scatter(data[lag_col], data[target_col], alpha=0.3, s=20, color='#2E86AB')

        # Add regression line if requested
        if add_regression:
            from scipy import stats
            x = data[lag_col].dropna()
            y = data.loc[x.index, target_col]
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            line_x = np.linspace(x.min(), x.max(), 100)
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'r--', linewidth=2,
                   label=f'R² = {r_value**2:.3f}')
            ax.legend()

        ax.set_xlabel(f'Lagged Demand ({lag_col})')
        ax.set_ylabel(f'Current Demand ({target_col})')
        ax.set_title(title)

        plt.tight_layout()
        self._save_figure(fig, filename, chapter)

    def plot_feature_target_relationships(self, data: pd.DataFrame,
                                          features: List[str],
                                          target_col: str,
                                          categorical_features: Optional[List[str]] = None,
                                          title: str = 'Feature-Target Relationships',
                                          filename: str = 'feature_target_relationships',
                                          chapter: str = 'chapter6',
                                          figsize: Tuple = (18, 24)):
        """
        Plot feature-target relationships (Figure 6.8).

        Uses scatter plots for numerical features and boxplots for categorical.

        Args:
            data: DataFrame with features and target
            features: List of feature column names
            target_col: Name of target column
            categorical_features: List of categorical feature names (use boxplot)
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Figure size
        """
        if categorical_features is None:
            categorical_features = []

        n_features = len(features)
        n_cols = min(2, n_features)  # Max 2 columns per row for better readability
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_features > 1 else [axes]

        for i, feature in enumerate(features):
            ax = axes[i]

            if feature in categorical_features:
                # Boxplot for categorical
                data_clean = data[[feature, target_col]].dropna()
                sns.boxplot(x=feature, y=target_col, data=data_clean, ax=ax, palette='Set2')
                # Rotate x-axis labels for long category names
                ax.tick_params(axis='x', rotation=90, labelsize=7)
                # Align rotated labels properly
                for label in ax.get_xticklabels():
                    label.set_ha('center')
            else:
                # Scatter plot for numerical
                ax.scatter(data[feature], data[target_col], alpha=0.3, s=15, color='#2E86AB')
                ax.tick_params(axis='x', rotation=0, labelsize=9)
                ax.grid(True, alpha=0.3)

            # Use FULL feature and target names - no truncation
            ax.set_xlabel(feature, fontsize=11, fontweight='bold')
            ax.set_ylabel(target_col, fontsize=11)
            ax.set_title(f'{feature} vs {target_col}', fontsize=10, fontweight='bold', pad=8)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # Place main title with more space above subplots
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)
        # Adjust spacing between subplots - more space for labels
        plt.subplots_adjust(hspace=0.5, wspace=0.3, top=0.94, bottom=0.08, left=0.08, right=0.95)
        self._save_figure(fig, filename, chapter)

    def plot_residual_time_series(self, time_index: np.ndarray,
                                   residuals_dict: Dict[str, np.ndarray],
                                   title: str,
                                   filename: str,
                                   chapter: str = 'chapter7',
                                   figsize: Tuple = (12, 6)):
        """
        Plot residual time series for multiple models (Figure 7.9).

        Args:
            time_index: Time index array (dates or integers)
            residuals_dict: Dict of {model_name: residuals_array}
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Figure size
        """
        n_models = len(residuals_dict)
        fig, axes = plt.subplots(n_models, 1, figsize=(figsize[0], figsize[1] * n_models / 2),
                                 sharex=True, sharey=True)

        if n_models == 1:
            axes = [axes]

        colors = ['#2E86AB', '#E94F37', '#44BBA4', '#393E41']

        for i, (model_name, residuals) in enumerate(residuals_dict.items()):
            ax = axes[i]
            ax.plot(time_index[:len(residuals)], residuals,
                   linewidth=1, color=colors[i % len(colors)], alpha=0.7)
            ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
            ax.set_ylabel(f'{model_name}\nResiduals')
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time Index')
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        self._save_figure(fig, filename, chapter)

    def plot_model_ranking_heatmap(self, results_df: pd.DataFrame,
                                    metric: str = 'RMSE',
                                    title: str = 'Model Ranking Heatmap',
                                    filename: str = 'model_ranking_heatmap',
                                    chapter: str = 'chapter7',
                                    figsize: Tuple = (12, 8)):
        """
        Plot model ranking heatmap across datasets (Figure 7.10).

        Args:
            results_df: DataFrame with 'model', 'case', and metric columns
            metric: Metric to rank by (lower is better for RMSE/MAE)
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Pivot to get models x cases matrix
        pivot = results_df.pivot_table(
            values=metric,
            index='model',
            columns='case',
            aggfunc='mean'
        )

        # Calculate ranks (1 = best)
        ranks = pivot.rank(axis=0, method='min')

        # Plot heatmap with ranks but annotate with metric values
        sns.heatmap(ranks, annot=pivot.round(2), fmt='',
                   cmap='RdYlGn_r', ax=ax,
                   cbar_kws={'label': f'Rank (by {metric})'})
        ax.set_title(title)
        ax.set_xlabel('Dataset/Case')
        ax.set_ylabel('Model')

        plt.tight_layout()
        self._save_figure(fig, filename, chapter)

    def plot_accuracy_vs_dataset_size(self, sizes: List[float],
                                       model_scores: Dict[str, List[float]],
                                       metric_name: str = 'RMSE',
                                       title: str = 'Accuracy vs Dataset Size',
                                       filename: str = 'accuracy_vs_size',
                                       chapter: str = 'chapter7',
                                       figsize: Tuple = DEFAULT_FIGSIZE):
        """
        Plot accuracy (RMSE) vs dataset size for multiple models (Figure 7.13).

        Args:
            sizes: List of dataset sizes (fractions: 0.25, 0.5, 0.75, 1.0)
            model_scores: Dict of {model_name: [scores_at_each_size]}
            metric_name: Name of the metric
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab10.colors
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

        for i, (model_name, scores) in enumerate(model_scores.items()):
            ax.plot(sizes, scores,
                   marker=markers[i % len(markers)],
                   linewidth=2, markersize=8,
                   color=colors[i % len(colors)],
                   label=model_name)

        ax.set_xlabel('Training Data Size (fraction)')
        ax.set_ylabel(metric_name)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        # Set x-axis to show percentages
        ax.set_xticks(sizes)
        ax.set_xticklabels([f'{int(s*100)}%' for s in sizes])

        plt.tight_layout()
        self._save_figure(fig, filename, chapter)

    def plot_peak_season_comparison(self, data: pd.DataFrame,
                                     time_col: str,
                                     actual_col: str,
                                     predictions: Dict[str, np.ndarray],
                                     peak_mask: np.ndarray,
                                     title: str,
                                     filename: str,
                                     chapter: str = 'chapter7',
                                     figsize: Tuple = (12, 6)):
        """
        Plot forecast vs actual during peak seasons (Figure 7.4).

        Args:
            data: DataFrame with actual values
            time_col: Time column name
            actual_col: Actual values column name
            predictions: Dict of {model_name: predictions_array}
            peak_mask: Boolean mask for peak periods
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Figure size
        """
        fig, ax = plt.subplots(figsize=figsize)

        peak_data = data[peak_mask].copy()
        colors = ['#E94F37', '#44BBA4', '#393E41']

        # Plot actual
        ax.plot(peak_data[time_col], peak_data[actual_col],
               label='Actual', linewidth=2, color='black', marker='o', markersize=4)

        # Plot predictions
        for i, (model_name, preds) in enumerate(predictions.items()):
            peak_preds = preds[peak_mask] if len(preds) == len(data) else preds[:len(peak_data)]
            ax.plot(peak_data[time_col].values[:len(peak_preds)], peak_preds,
                   label=model_name, linewidth=1.5, linestyle='--',
                   color=colors[i % len(colors)], marker='s', markersize=3)

        ax.set_xlabel('Time')
        ax.set_ylabel(actual_col)
        ax.set_title(title)
        ax.legend()
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        self._save_figure(fig, filename, chapter)

    def plot_model_config_table(self, params: Dict[str, str],
                                title: str,
                                filename: str,
                                chapter: str = 'chapter7',
                                figsize: Optional[Tuple] = None):
        """
        Plot a table figure with model configuration parameters.

        Args:
            params: Dict of {param_name: param_value}
            title: Figure title
            filename: Output filename
            chapter: Output chapter directory
            figsize: Optional figure size (auto-scaled if None)
        """
        rows = [[k, v] for k, v in params.items()]
        df = pd.DataFrame(rows, columns=['Parameter', 'Value'])

        n_rows = max(1, len(df))
        if figsize is None:
            figsize = (8, max(2.5, 0.35 * n_rows))

        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc='center',
            cellLoc='left',
            colLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)

        ax.set_title(title, fontsize=14, pad=12)
        plt.tight_layout()
        self._save_figure(fig, filename, chapter)
