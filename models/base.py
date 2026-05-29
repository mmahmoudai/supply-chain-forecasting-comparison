"""Base model class for all forecasting models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import logging


class BaseModel(ABC):
    """Abstract base class for all forecasting models."""
    
    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.model = None
        self.is_fitted = False
        self.logger = logging.getLogger(self.__class__.__name__)
        self.training_history = {}
    
    @abstractmethod
    def fit(self, X: Union[pd.DataFrame, np.ndarray], 
            y: Union[pd.Series, np.ndarray], **kwargs) -> 'BaseModel':
        """
        Fit the model to training data.
        
        Args:
            X: Features (or time series for univariate models)
            y: Target values
            **kwargs: Additional model-specific parameters
            
        Returns:
            self for chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray, int], 
                **kwargs) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            X: Features or number of steps to forecast
            **kwargs: Additional parameters
            
        Returns:
            Array of predictions
        """
        pass
    
    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        self.logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BaseModel':
        """
        Load a model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {}
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class TimeSeriesModel(BaseModel):
    """Base class for univariate time series models."""
    
    def __init__(self, name: str = "TimeSeriesModel"):
        super().__init__(name)
        self.fitted_values = None
        self.residuals = None
    
    def fit(self, y: Union[pd.Series, np.ndarray], **kwargs) -> 'TimeSeriesModel':
        """
        Fit the time series model.
        
        Args:
            y: Time series data
            **kwargs: Additional parameters
            
        Returns:
            self for chaining
        """
        pass
    
    def forecast(self, steps: int, **kwargs) -> np.ndarray:
        """
        Forecast future values.
        
        Args:
            steps: Number of steps to forecast
            **kwargs: Additional parameters
            
        Returns:
            Array of forecasted values
        """
        return self.predict(steps, **kwargs)


class SupervisedModel(BaseModel):
    """Base class for supervised learning models."""
    
    def __init__(self, name: str = "SupervisedModel"):
        super().__init__(name)
        self.feature_names = None
        self.feature_importance = None
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if self.feature_importance is None:
            return None
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
