"""Exponential Smoothing (ETS) model."""

import numpy as np
import pandas as pd
from typing import Union, Optional, Literal
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.base import TimeSeriesModel


class ETSModel(TimeSeriesModel):
    """
    Exponential Smoothing (ETS) model using statsmodels.
    
    Supports additive/multiplicative trend and seasonality.
    """
    
    def __init__(self,
                 trend: Optional[Literal['add', 'mul']] = 'add',
                 seasonal: Optional[Literal['add', 'mul']] = None,
                 seasonal_periods: Optional[int] = None,
                 damped_trend: bool = False,
                 use_boxcox: bool = False):
        """
        Initialize the ETS model.
        
        Args:
            trend: Type of trend component ('add', 'mul', or None)
            seasonal: Type of seasonal component ('add', 'mul', or None)
            seasonal_periods: Number of periods in a complete seasonal cycle
            damped_trend: Whether to dampen the trend
            use_boxcox: Whether to use Box-Cox transformation
        """
        super().__init__("ETS")
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.use_boxcox = use_boxcox
        self._results = None
    
    def fit(self, y: Union[pd.Series, np.ndarray], **kwargs) -> 'ETSModel':
        """
        Fit the ETS model.
        
        Args:
            y: Time series data
            
        Returns:
            self for chaining
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        # Ensure positive values for multiplicative components
        if self.trend == 'mul' or self.seasonal == 'mul':
            if (y <= 0).any():
                y = y - y.min() + 1
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                model = ExponentialSmoothing(
                    y,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods,
                    damped_trend=self.damped_trend,
                    use_boxcox=self.use_boxcox
                )
                self._results = model.fit(optimized=True)
                self.model = model
                
            except Exception as e:
                self.logger.warning(f"ETS fitting failed: {e}. Trying simpler model.")
                # Fall back to simpler model
                model = ExponentialSmoothing(
                    y,
                    trend='add',
                    seasonal=None,
                    damped_trend=False
                )
                self._results = model.fit(optimized=True)
                self.model = model
        
        self.fitted_values = self._results.fittedvalues.values
        self.residuals = y.values - self.fitted_values
        self.is_fitted = True
        
        self.logger.info(f"ETS fitted with AIC: {self._results.aic:.2f}")
        return self
    
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Array of forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        forecasts = self._results.forecast(steps)
        return forecasts.values
    
    def get_params(self) -> dict:
        """Get model parameters."""
        params = {
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'damped_trend': self.damped_trend
        }
        
        if self._results is not None:
            params['aic'] = self._results.aic
            params['bic'] = self._results.bic
        
        return params
    
    @staticmethod
    def auto_fit(y: Union[pd.Series, np.ndarray],
                 seasonal_periods: Optional[int] = None) -> 'ETSModel':
        """
        Automatically select the best ETS model based on AIC.
        
        Args:
            y: Time series data
            seasonal_periods: Seasonal period to try
            
        Returns:
            Best fitted ETS model
        """
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        trend_options = ['add', None]
        seasonal_options = ['add', None] if seasonal_periods else [None]
        damped_options = [True, False]
        
        best_model = None
        best_aic = np.inf
        
        for trend in trend_options:
            for seasonal in seasonal_options:
                for damped in damped_options:
                    if damped and trend is None:
                        continue
                    
                    try:
                        model = ETSModel(
                            trend=trend,
                            seasonal=seasonal,
                            seasonal_periods=seasonal_periods,
                            damped_trend=damped
                        )
                        model.fit(y)
                        
                        if model._results.aic < best_aic:
                            best_aic = model._results.aic
                            best_model = model
                            
                    except Exception:
                        continue
        
        if best_model is None:
            # Fall back to simple model
            best_model = ETSModel(trend='add', seasonal=None)
            best_model.fit(y)
        
        return best_model
