"""SARIMAX (Seasonal ARIMA with eXogenous variables) model implementation."""

import numpy as np
import pandas as pd
from typing import Union, Optional, Tuple
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.base import TimeSeriesModel


class SARIMAXModel(TimeSeriesModel):
    """
    SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables).
    
    Extends ARIMAX with seasonal components and optional exogenous regressors.
    """
    
    def __init__(self,
                 order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 4),
                 auto_order: bool = False,
                 max_p: int = 3,
                 max_d: int = 2,
                 max_q: int = 3,
                 max_P: int = 2,
                 max_D: int = 1,
                 max_Q: int = 2):
        """
        Initialize the SARIMAX model.
        
        Args:
            order: (p, d, q) non-seasonal order
            seasonal_order: (P, D, Q, s) seasonal order
            auto_order: Whether to automatically select orders
            max_*: Maximum orders for auto selection
        """
        super().__init__("SARIMAX")
        self.order = order
        self.seasonal_order = seasonal_order
        self.auto_order = auto_order
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.max_P = max_P
        self.max_D = max_D
        self.max_Q = max_Q
        self._results = None
        self._y_train = None
        self._exog_train = None
    
    def _auto_select_order(self, y: np.ndarray, s: int,
                           exog=None) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """
        Automatically select SARIMAX orders using AIC.
        
        Args:
            y: Time series data
            s: Seasonal period
            exog: Exogenous variables (optional)
            
        Returns:
            Tuple of (order, seasonal_order)
        """
        best_aic = np.inf
        best_order = (1, 1, 1)
        best_seasonal = (1, 1, 1, s)
        
        # Reduced search space for efficiency
        p_range = range(min(3, self.max_p + 1))
        q_range = range(min(3, self.max_q + 1))
        P_range = range(min(2, self.max_P + 1))
        Q_range = range(min(2, self.max_Q + 1))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for p in p_range:
                for q in q_range:
                    for P in P_range:
                        for Q in Q_range:
                            if p == 0 and q == 0 and P == 0 and Q == 0:
                                continue
                            
                            try:
                                model = SARIMAX(
                                    y, exog=exog,
                                    order=(p, 1, q),
                                    seasonal_order=(P, 1, Q, s),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )
                                results = model.fit(disp=False, maxiter=50)
                                
                                if results.aic < best_aic:
                                    best_aic = results.aic
                                    best_order = (p, 1, q)
                                    best_seasonal = (P, 1, Q, s)
                                    
                            except Exception:
                                continue
        
        self.logger.info(f"Auto-selected SARIMAX order: {best_order} x {best_seasonal} with AIC: {best_aic:.2f}")
        return best_order, best_seasonal
    
    def fit(self, y: Union[pd.Series, np.ndarray],
            exog: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            seasonal_period: Optional[int] = None,
            **kwargs) -> 'SARIMAXModel':
        """
        Fit the SARIMAX model.
        
        Args:
            y: Time series data
            exog: Exogenous variables (optional)
            seasonal_period: Override seasonal period
            
        Returns:
            self for chaining
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        if isinstance(exog, pd.DataFrame):
            exog = exog.values
        
        self._y_train = y.copy()
        self._exog_train = exog
        
        # Get seasonal period
        s = seasonal_period if seasonal_period else self.seasonal_order[3]
        
        # Auto-select order if requested
        if self.auto_order:
            self.order, self.seasonal_order = self._auto_select_order(y, s, exog)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                model = SARIMAX(
                    y, exog=exog,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self._results = model.fit(disp=False, maxiter=200)
                self.model = model
                
            except Exception as e:
                self.logger.warning(f"SARIMAX fitting failed: {e}. Trying simpler model.")
                model = SARIMAX(
                    y, exog=exog,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, s),
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                self._results = model.fit(disp=False, maxiter=100)
                self.model = model
                self.order = (1, 1, 1)
                self.seasonal_order = (1, 1, 1, s)
        
        self.fitted_values = self._results.fittedvalues
        self.residuals = self._results.resid
        self.is_fitted = True
        
        self.logger.info(f"SARIMAX{self.order}x{self.seasonal_order} fitted with AIC: {self._results.aic:.2f}")
        return self
    
    def predict(self, steps: int,
                exog_future: Optional[Union[pd.DataFrame, np.ndarray]] = None,
                **kwargs) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            exog_future: Exogenous variables for forecast horizon
            
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
            'seasonal_order': self.seasonal_order,
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
