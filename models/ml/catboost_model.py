"""CatBoost Regressor model."""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict
from catboost import CatBoostRegressor
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.base import SupervisedModel
from config.constants import RANDOM_SEED, CATBOOST_DEFAULTS


class CatBoostModel(SupervisedModel):
    """
    CatBoost Regressor with native categorical feature support.
    
    Uses Ordered Target Statistics and symmetric tree structures
    for gradient boosting with reduced prediction shift.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the CatBoost model.
        
        Args:
            **kwargs: Override default hyperparameters
        """
        super().__init__("CatBoost")
        
        # Merge defaults with provided kwargs
        self.params = CATBOOST_DEFAULTS.copy()
        self.params.update(kwargs)
        
        self.model = CatBoostRegressor(**self.params)
        self.best_params = None
        self.eval_results = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_val: Optional[Union[pd.Series, np.ndarray]] = None,
            early_stopping_rounds: int = 50,
            cat_features: Optional[list] = None,
            **kwargs) -> 'CatBoostModel':
        """
        Fit the CatBoost model.
        
        Args:
            X: Feature matrix (DataFrame preferred for categorical support)
            y: Target values
            X_val: Validation features (for early stopping)
            y_val: Validation targets
            early_stopping_rounds: Rounds for early stopping
            cat_features: Names or indices of categorical features.
                          If None, auto-detects from object/category dtype columns.
            
        Returns:
            self for chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            
            # Auto-detect categorical features if not provided
            if cat_features is None:
                cat_features = [c for c in X.columns 
                               if X[c].dtype in ['object', 'category']]
            
            # Keep as DataFrame for CatBoost native categorical handling
            if cat_features:
                X_fit = X.copy()
                for c in cat_features:
                    if c in X_fit.columns:
                        X_fit[c] = X_fit[c].astype(str)
                self.logger.info(f"CatBoost using {len(cat_features)} native categorical features: {cat_features}")
            else:
                X_fit = X.values
        else:
            if self.feature_names is None:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            X_fit = X
            cat_features = None  # Can't auto-detect from numpy arrays
        
        if isinstance(y, pd.Series):
            y = y.values
        
        # Prepare evaluation set for early stopping
        eval_set = None
        if X_val is not None and y_val is not None:
            if isinstance(X_val, pd.DataFrame) and cat_features:
                X_val_fit = X_val.copy()
                for c in cat_features:
                    if c in X_val_fit.columns:
                        X_val_fit[c] = X_val_fit[c].astype(str)
            elif isinstance(X_val, pd.DataFrame):
                X_val_fit = X_val.values
            else:
                X_val_fit = X_val
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            eval_set = (X_val_fit, y_val)
        
        # Fit with or without early stopping
        fit_params = {
            'X': X_fit,
            'y': y,
            'verbose': False
        }
        
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
            fit_params['early_stopping_rounds'] = early_stopping_rounds
        
        if cat_features:
            fit_params['cat_features'] = cat_features
        
        self.model.fit(**fit_params)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        self.is_fitted = True
        
        # Store best iteration if early stopping was used
        if hasattr(self.model, 'best_iteration_') and self.model.best_iteration_ is not None:
            self.training_history['best_iteration'] = self.model.best_iteration_
        
        # Calculate training R-squared
        train_pred = self.model.predict(X_fit)
        ss_res = np.sum((y - train_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        self.training_history['r2_train'] = r2
        self.logger.info(f"CatBoost fitted with R² = {r2:.4f}")
        
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        return self.model.predict(X)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'params': self.params,
            'best_params': self.best_params,
            'best_iteration': self.training_history.get('best_iteration')
        }
    
    def get_feature_importance(self, top_n: int = 20,
                                feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            top_n: Number of top features to return
            feature_names: Optional list of feature names (overrides stored names)

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Use provided feature names or stored ones
        names = feature_names if feature_names is not None else self.feature_names

        importance = self.model.feature_importances_

        importance_list = []
        for i, name in enumerate(names):
            score = importance[i] if i < len(importance) else 0
            importance_list.append({'feature': name, 'importance': score})

        importance_df = pd.DataFrame(importance_list)
        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df.head(top_n)
