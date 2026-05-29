"""Naive and Seasonal Naive forecasting models."""

import numpy as np
import pandas as pd
from typing import Union, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.base import TimeSeriesModel


class NaiveForecaster(TimeSeriesModel):
    """
    Naive and Seasonal Naive forecaster.
    
    Naive: Uses the last observation as forecast.
    Seasonal Naive: Uses the observation from the same period last season.
    """
    
    def __init__(self, seasonal: bool = False, seasonal_period: int = 7):
        """
        Initialize the Naive Forecaster.
        
        Args:
            seasonal: Whether to use seasonal naive method
            seasonal_period: Number of periods in a season (e.g., 7 for weekly)
        """
        name = "SeasonalNaive" if seasonal else "Naive"
        super().__init__(name)
        self.seasonal = seasonal
        self.seasonal_period = seasonal_period
        self._last_values = None
        self._history = None
    
    def fit(self, y: Union[pd.Series, np.ndarray], **kwargs) -> 'NaiveForecaster':
        """
        Fit the naive forecaster (stores necessary history).
        
        Args:
            y: Time series data
            
        Returns:
            self for chaining
        """
        if isinstance(y, pd.Series):
            y = y.values
        
        self._history = y.copy()
        
        if self.seasonal:
            # Store the last `seasonal_period` values for seasonal naive
            self._last_values = y[-self.seasonal_period:]
        else:
            # Store just the last value for simple naive
            self._last_values = np.array([y[-1]])
        
        self.fitted_values = np.concatenate([[np.nan], y[:-1]])
        self.residuals = y - self.fitted_values
        self.is_fitted = True
        
        self.logger.info(f"{self.name} fitted on {len(y)} observations")
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
        
        if self.seasonal:
            # Seasonal naive: repeat the last season
            forecasts = np.tile(self._last_values, 
                               int(np.ceil(steps / self.seasonal_period)))[:steps]
        else:
            # Simple naive: repeat the last value
            forecasts = np.full(steps, self._last_values[0])
        
        return forecasts
    
    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'seasonal': self.seasonal,
            'seasonal_period': self.seasonal_period
        }
