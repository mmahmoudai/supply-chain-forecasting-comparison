"""ARIMAX model implementation using SARIMAX with exogenous support."""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import warnings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.base import TimeSeriesModel


class ARIMAXModel(TimeSeriesModel):
    """
    ARIMAX (AutoRegressive Integrated Moving Average with eXogenous variables).
    
    Uses statsmodels SARIMAX backend with seasonal_order=(0,0,0,0) to
    implement a non-seasonal ARIMAX model with optional exogenous regressors.
    """
    
    def __init__(self, 
                 order: Tuple[int, int, int] = (1, 1, 1),
                 auto_order: bool = False,
                 max_p: int = 5,
                 max_d: int = 2,
                 max_q: int = 5):
        """
        Initialize the ARIMAX model.
        
        Args:
            order: (p, d, q) order of the ARIMAX model
            auto_order: Whether to automatically select the order
            max_p: Maximum AR order for auto selection
            max_d: Maximum differencing order for auto selection
            max_q: Maximum MA order for auto selection
        """
        super().__init__("ARIMAX")
        self.order = order
        self.auto_order = auto_order
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self._results = None
        self._y_train = None
        self._exog_train = None
    
    def _determine_d(self, y: np.ndarray) -> int:
        """
        Determine the differencing order using ADF test.
        
        Args:
            y: Time series data
            
        Returns:
            Recommended differencing order
        """
        d = 0
        y_diff = y.copy()
        
        for i in range(self.max_d + 1):
            result = adfuller(y_diff, autolag='AIC')
            if result[1] < 0.05:  # Stationary at 5% significance
                return d
            if d < self.max_d:
                y_diff = np.diff(y_diff)
                d += 1
        
        return min(d, self.max_d)
    
    def _auto_select_order(self, y: np.ndarray, exog=None) -> Tuple[int, int, int]:
        """
        Automatically select ARIMAX order using AIC.
        
        Args:
            y: Time series data
            exog: Exogenous variables (optional)
            
        Returns:
            Best (p, d, q) order
        """
        d = self._determine_d(y)
        best_aic = np.inf
        best_order = (1, d, 1)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for p in range(self.max_p + 1):
                for q in range(self.max_q + 1):
                    if p == 0 and q == 0:
                        continue
                    
                    try:
                        model = SARIMAX(
                            y, exog=exog,
                            order=(p, d, q),
                            seasonal_order=(0, 0, 0, 0),
                            enforce_stationarity=False,
                            enforce_invertibility=False
                        )
                        results = model.fit(disp=False, maxiter=50)
                        
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_order = (p, d, q)
                            
                    except Exception:
                        continue
        
        self.logger.info(f"Auto-selected ARIMAX order: {best_order} with AIC: {best_aic:.2f}")
        return best_order
    
    def fit(self, y: Union[pd.Series, np.ndarray],
            exog: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            **kwargs) -> 'ARIMAXModel':
        """
        Fit the ARIMAX model.
        
        Args:
            y: Time series data
            exog: Exogenous variables (optional)
            
        Returns:
            self for chaining
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        if isinstance(exog, pd.DataFrame):
            exog = exog.values
        
        self._y_train = y.copy()
        self._exog_train = exog
        
        # Auto-select order if requested
        if self.auto_order:
            self.order = self._auto_select_order(y, exog)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                model = SARIMAX(
                    y, exog=exog,
                    order=self.order,
                    seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self._results = model.fit(disp=False, maxiter=200)
                self.model = model
                
            except Exception as e:
                self.logger.warning(f"ARIMAX({self.order}) failed: {e}. Trying (1,1,1).")
                model = SARIMAX(
                    y, exog=exog,
                    order=(1, 1, 1),
                    seasonal_order=(0, 0, 0, 0),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self._results = model.fit(disp=False, maxiter=200)
                self.model = model
                self.order = (1, 1, 1)
        
        self.fitted_values = self._results.fittedvalues
        self.residuals = self._results.resid
        self.is_fitted = True
        
        self.logger.info(f"ARIMAX{self.order} fitted with AIC: {self._results.aic:.2f}")
        return self
    
    def predict(self, steps: int,
                exog_future: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                **kwargs) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            exog_future: Exogenous variables for forecast horizon (required if exog was used in fit)
            
        Returns:
            Array of forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(exog_future, pd.DataFrame):
            exog_future = exog_future.values
        
        forecasts = self._results.forecast(steps, exog=exog_future)
        
        if isinstance(forecasts, pd.Series):
            forecasts = forecasts.values
        
        return forecasts
    
    def get_params(self) -> dict:
        """Get model parameters."""
        params = {
            'order': self.order,
            'auto_order': self.auto_order,
            'has_exog': self._exog_train is not None
        }
        
        if self._results is not None:
            params['aic'] = self._results.aic
            params['bic'] = self._results.bic
        
        return params
    
    def summary(self) -> str:
        """Get model summary."""
        if self._results is None:
            return "Model not fitted"
        return str(self._results.summary())
