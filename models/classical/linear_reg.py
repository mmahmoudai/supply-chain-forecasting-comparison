"""Linear Regression model for supervised learning."""

import numpy as np
import pandas as pd
from typing import Union, Optional, List
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.base import SupervisedModel


class LinearRegressionModel(SupervisedModel):
    """
    Linear Regression model with optional regularization.
    
    Supports standard OLS, Ridge (L2), and Lasso (L1) regression.
    """
    
    def __init__(self,
                 regularization: Optional[str] = None,
                 alpha: float = 1.0,
                 fit_intercept: bool = True):
        """
        Initialize the Linear Regression model.
        
        Args:
            regularization: Type of regularization ('ridge', 'lasso', or None)
            alpha: Regularization strength (for Ridge/Lasso)
            fit_intercept: Whether to fit an intercept term
        """
        name = regularization.capitalize() if regularization else "LinearRegression"
        super().__init__(name)
        self.regularization = regularization
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self._create_model()
    
    def _create_model(self):
        """Create the appropriate sklearn model."""
        if self.regularization == 'ridge':
            self.model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept)
        elif self.regularization == 'lasso':
            self.model = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept)
        else:
            self.model = LinearRegression(fit_intercept=self.fit_intercept)
    
    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray], **kwargs) -> 'LinearRegressionModel':
        """
        Fit the linear regression model.
        
        Args:
            X: Feature matrix
            y: Target values
            
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
        
        self.model.fit(X, y)
        
        # Store coefficients as feature importance
        self.feature_importance = np.abs(self.model.coef_)
        self.is_fitted = True
        
        # Calculate R-squared on training data
        train_pred = self.model.predict(X)
        ss_res = np.sum((y - train_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        self.training_history['r2_train'] = r2
        self.logger.info(f"Linear Regression fitted with R² = {r2:.4f}")
        
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
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get regression coefficients.
        
        Returns:
            DataFrame with feature names and coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_
        })
        
        if self.fit_intercept:
            intercept_row = pd.DataFrame({
                'feature': ['intercept'],
                'coefficient': [self.model.intercept_]
            })
            coef_df = pd.concat([intercept_row, coef_df], ignore_index=True)
        
        return coef_df.sort_values('coefficient', key=abs, ascending=False)
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'regularization': self.regularization,
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'coefficients': self.model.coef_.tolist() if self.is_fitted else None,
            'intercept': self.model.intercept_ if self.is_fitted else None
        }
