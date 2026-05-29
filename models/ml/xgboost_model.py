"""XGBoost Regressor model."""

import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Tuple
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.base import SupervisedModel
from config.constants import RANDOM_SEED, XGBOOST_DEFAULTS


class XGBoostModel(SupervisedModel):
    """
    XGBoost Regressor with early stopping and hyperparameter tuning.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the XGBoost model.
        
        Args:
            **kwargs: Override default hyperparameters
        """
        super().__init__("XGBoost")
        
        # Merge defaults with provided kwargs
        self.params = XGBOOST_DEFAULTS.copy()
        self.params.update(kwargs)
        
        self.model = xgb.XGBRegressor(**self.params)
        self.best_params = None
        self.eval_results = None
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_val: Optional[Union[pd.Series, np.ndarray]] = None,
            early_stopping_rounds: int = 50,
            tune_hyperparams: bool = False,
            param_grid: Optional[Dict] = None,
            cv: int = 5,
            n_iter: int = 20,
            **kwargs) -> 'XGBoostModel':
        """
        Fit the XGBoost model.
        
        Args:
            X: Feature matrix
            y: Target values
            X_val: Validation features (for early stopping)
            y_val: Validation targets
            early_stopping_rounds: Rounds for early stopping
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
            # Prepare evaluation set for early stopping
            eval_set = None
            if X_val is not None and y_val is not None:
                if isinstance(X_val, pd.DataFrame):
                    X_val = X_val.values
                if isinstance(y_val, pd.Series):
                    y_val = y_val.values
                eval_set = [(X_val, y_val)]
            
            # Fit with or without early stopping
            if eval_set is not None:
                self.model.fit(
                    X, y,
                    eval_set=eval_set,
                    verbose=False
                )
                # Store best iteration if early stopping was used
                if hasattr(self.model, 'best_iteration'):
                    self.training_history['best_iteration'] = self.model.best_iteration
            else:
                self.model.fit(X, y, verbose=False)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        self.is_fitted = True
        
        # Calculate training R-squared
        train_pred = self.model.predict(X)
        ss_res = np.sum((y - train_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        self.training_history['r2_train'] = r2
        self.logger.info(f"XGBoost fitted with R² = {r2:.4f}")
        
        return self
    
    def _tune_hyperparameters(self, X: np.ndarray, y: np.ndarray,
                               param_grid: Optional[Dict],
                               cv: int, n_iter: int):
        """Perform hyperparameter tuning using RandomizedSearchCV."""
        
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200, 300, 500],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_child_weight': [1, 3, 5, 7],
                'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
                'reg_alpha': [0, 0.1, 0.5, 1],
                'reg_lambda': [0, 0.1, 0.5, 1]
            }
        
        base_model = xgb.XGBRegressor(random_state=RANDOM_SEED, verbosity=0)
        
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
            'best_iteration': self.training_history.get('best_iteration')
        }
    
    def get_feature_importance(self, top_n: int = 20,
                                importance_type: str = 'weight',
                                feature_names: Optional[list] = None) -> pd.DataFrame:
        """
        Get feature importance.

        Args:
            top_n: Number of top features to return
            importance_type: Type of importance ('weight', 'gain', 'cover')
            feature_names: Optional list of feature names (overrides stored names)

        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Use provided feature names or stored ones
        names = feature_names if feature_names is not None else self.feature_names

        # Get importance scores
        booster = self.model.get_booster()
        importance = booster.get_score(importance_type=importance_type)

        # Map to feature names
        importance_list = []
        for i, name in enumerate(names):
            feat_key = f'f{i}'
            score = importance.get(feat_key, 0)
            importance_list.append({'feature': name, 'importance': score})

        importance_df = pd.DataFrame(importance_list)
        importance_df = importance_df.sort_values('importance', ascending=False)

        return importance_df.head(top_n)
