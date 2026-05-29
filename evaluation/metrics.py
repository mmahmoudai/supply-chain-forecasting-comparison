"""Evaluation metrics for regression models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import logging


class MetricsCalculator:
    """Calculate and compare evaluation metrics for regression models."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Calculate Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            epsilon: Small value to avoid division by zero
            
        Returns:
            MAPE as a percentage
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Avoid division by zero
        mask = np.abs(y_true) > epsilon
        if not mask.any():
            return np.nan
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared (coefficient of determination)."""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Root Mean Squared Logarithmic Error.
        
        Used for Case 2 evaluation.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Ensure non-negative values
        y_true = np.maximum(y_true, 0)
        y_pred = np.maximum(y_pred, 0)
        
        log_true = np.log1p(y_true)
        log_pred = np.log1p(y_pred)
        
        return np.sqrt(np.mean((log_true - log_pred) ** 2))
    
    def calculate_all_metrics(self, y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               include_rmsle: bool = False) -> Dict[str, float]:
        """
        Calculate all regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            include_rmsle: Whether to include RMSLE (for Case 2)
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {
            'MAE': self.mae(y_true, y_pred),
            'RMSE': self.rmse(y_true, y_pred),
            'MAPE': self.mape(y_true, y_pred),
            'R2': self.r2(y_true, y_pred)
        }
        
        if include_rmsle:
            metrics['RMSLE'] = self.rmsle(y_true, y_pred)
        
        return metrics
    
    def evaluate_model(self, model_name: str,
                       y_true: np.ndarray,
                       y_pred: np.ndarray,
                       case: str,
                       dataset_split: str = 'test',
                       include_rmsle: bool = False) -> Dict[str, float]:
        """
        Evaluate a model and store results.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            case: Case identifier (case1, case2, etc.)
            dataset_split: 'train', 'val', or 'test'
            include_rmsle: Whether to include RMSLE
            
        Returns:
            Dictionary of metrics
        """
        metrics = self.calculate_all_metrics(y_true, y_pred, include_rmsle)
        
        # Store results
        key = f"{case}_{model_name}_{dataset_split}"
        self.results[key] = {
            'case': case,
            'model': model_name,
            'split': dataset_split,
            **metrics
        }
        
        self.logger.info(f"{model_name} on {case} ({dataset_split}): "
                        f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
                        f"R2={metrics['R2']:.4f}")
        
        return metrics
    
    def compare_models(self, case: Optional[str] = None,
                       split: str = 'test') -> pd.DataFrame:
        """
        Compare all evaluated models.
        
        Args:
            case: Filter by case (optional)
            split: Filter by dataset split
            
        Returns:
            DataFrame with model comparison
        """
        results_list = []
        
        for key, result in self.results.items():
            if split and result['split'] != split:
                continue
            if case and result['case'] != case:
                continue
            results_list.append(result)
        
        if not results_list:
            return pd.DataFrame()
        
        df = pd.DataFrame(results_list)
        
        # Sort by RMSE (lower is better)
        df = df.sort_values('RMSE')
        
        return df
    
    def get_best_model(self, case: str, 
                       metric: str = 'RMSE',
                       split: str = 'test') -> str:
        """
        Get the best performing model for a case.
        
        Args:
            case: Case identifier
            metric: Metric to use for comparison
            split: Dataset split to compare
            
        Returns:
            Name of the best model
        """
        comparison = self.compare_models(case=case, split=split)
        
        if comparison.empty:
            return None
        
        # R2 is maximized, others are minimized
        if metric == 'R2':
            best_idx = comparison[metric].idxmax()
        else:
            best_idx = comparison[metric].idxmin()
        
        return comparison.loc[best_idx, 'model']
    
    def create_summary_table(self) -> pd.DataFrame:
        """
        Create a summary table of all results.
        
        Returns:
            Pivot table with models as rows and cases as columns
        """
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(list(self.results.values()))
        
        # Filter to test results
        df_test = df[df['split'] == 'test']
        
        # Create pivot table
        pivot = df_test.pivot_table(
            values=['MAE', 'RMSE', 'R2'],
            index='model',
            columns='case',
            aggfunc='mean'
        )
        
        return pivot
    
    def to_csv(self, filepath: str):
        """Save results to CSV file."""
        df = pd.DataFrame(list(self.results.values()))
        df.to_csv(filepath, index=False)
        self.logger.info(f"Results saved to {filepath}")
    
    def to_latex(self, case: Optional[str] = None) -> str:
        """
        Generate LaTeX table for thesis.
        
        Args:
            case: Filter by case (optional)
            
        Returns:
            LaTeX formatted table string
        """
        comparison = self.compare_models(case=case, split='test')
        
        if comparison.empty:
            return ""
        
        # Select relevant columns
        cols = ['model', 'MAE', 'RMSE', 'MAPE', 'R2']
        if 'RMSLE' in comparison.columns:
            cols.append('RMSLE')
        
        df = comparison[cols].copy()
        
        # Format numbers
        for col in ['MAE', 'RMSE', 'MAPE']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f'{x:.4f}')
        df['R2'] = df['R2'].apply(lambda x: f'{x:.4f}')
        
        return df.to_latex(index=False, escape=False)
