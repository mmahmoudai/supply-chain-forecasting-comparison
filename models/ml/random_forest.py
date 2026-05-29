"""Random Forest Regressor model."""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.base import SupervisedModel
from config.constants import RANDOM_SEED, RANDOM_FOREST_DEFAULTS


class RandomForestModel(SupervisedModel):
    """
    Random Forest Regressor with hyperparameter tuning support.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Random Forest model.
        
        Args:
            **kwargs: Override default hyperparameters
        """
        super().__init__("RandomForest")
        
        # Merge defaults with provided kwargs
        self.params = RANDOM_FOREST_DEFAULTS.copy()
        self.params.update(kwargs)
        
        self.model = RandomForestRegressor(**self.params)
        self.best_params = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            tune_hyperparams: bool = False,
            param_grid: Optional[Dict] = None,
            cv: int = 5,
            n_iter: int = 20,
            **kwargs) -> 'RandomForestModel':
        """
        Fit the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target values
            tune_hyperparams: Whether to perform hyperparameter tuning
            param_grid: Parameter grid for tuning
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random search
            
        Returns:
            self for chaining
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        elif self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        if isinstance(y, pd.Series):
            y = y.values
        
        if tune_hyperparams:
            self._tune_hyperparameters(X, y, param_grid, cv, n_iter)
        else:
            self.model.fit(X, y)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        self.is_fitted = True
        
        # Calculate training R-squared
        train_pred = self.model.predict(X)
        ss_res = np.sum((y - train_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        self.training_history['r2_train'] = r2
        self.logger.info(f"Random Forest fitted with R² = {r2:.4f}")
        
        return self
    
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                               param_grid: Optional[Dict],
                               cv: int, n_iter: int):
        """Perform hyperparameter tuning using RandomizedSearchCV."""
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
        
        base_model = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1)
        
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=RANDOM_SEED,
            n_jobs=-1,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        self.best_params = random_search.best_params_
        self.model = random_search.best_estimator_
        self.params.update(self.best_params)
        
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best CV score: {-random_search.best_score_:.4f} (MSE)")
    
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
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth
        }
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
