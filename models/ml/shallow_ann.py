"""Shallow ANN regressor using PyTorch with GPU support."""

import numpy as np
import pandas as pd
from typing import Union, Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.base import SupervisedModel
from config.constants import RANDOM_SEED, SHALLOW_ANN_DEFAULTS

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


def _get_device():
    """Get the best available device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class _ShallowNetwork(nn.Module):
    """PyTorch single hidden layer network."""

    def __init__(self, input_dim: int, hidden_units: int,
                 activation: str = 'relu', dropout_rate: float = 0.1):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'tanh':
            layers.append(nn.Tanh())
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(hidden_units, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class ShallowANNModel(SupervisedModel):
    """
    Shallow (single hidden layer) ANN for regression using PyTorch.

    Uses GPU acceleration when available (CUDA).
    """

    def __init__(self,
                 hidden_units: Optional[int] = None,
                 activation: Optional[str] = None,
                 learning_rate: Optional[float] = None,
                 batch_size: Optional[int] = None,
                 epochs: Optional[int] = None,
                 patience: Optional[int] = None,
                 dropout_rate: Optional[float] = None,
                 scale_inputs: Optional[bool] = None,
                 scale_target: Optional[bool] = None,
                 **kwargs):
        super().__init__("ShallowANN")

        self.device = _get_device()

        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)

        defaults = SHALLOW_ANN_DEFAULTS
        self.hidden_units = hidden_units if hidden_units is not None else defaults['hidden_units']
        self.activation = activation if activation is not None else defaults['activation']
        self.dropout_rate = dropout_rate if dropout_rate is not None else defaults['dropout_rate']
        self.learning_rate = learning_rate if learning_rate is not None else defaults['learning_rate']
        self.batch_size = batch_size if batch_size is not None else defaults['batch_size']
        self.epochs = epochs if epochs is not None else defaults['epochs']
        self.patience = patience if patience is not None else defaults['patience']
        self.scale_inputs = scale_inputs if scale_inputs is not None else defaults['scale_inputs']
        self.scale_target = scale_target if scale_target is not None else defaults['scale_target']

        self.history = {'loss': [], 'val_loss': []}
        self.scaler_X = None
        self.scaler_y = None

    def _prepare_features(self, X, fit=False):
        if isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                self.feature_names = X.columns.tolist()
            X = X.values

        if not self.scale_inputs:
            return X

        if fit or self.scaler_X is None:
            self.scaler_X = StandardScaler()
            return self.scaler_X.fit_transform(X)
        return self.scaler_X.transform(X)

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_val: Optional[Union[pd.Series, np.ndarray]] = None,
            verbose: int = 0,
            **kwargs) -> 'ShallowANNModel':
        """Fit the shallow ANN model."""
        if isinstance(X, pd.DataFrame) and self.feature_names is None:
            self.feature_names = X.columns.tolist()

        if isinstance(y, pd.Series):
            y = y.values
        y = y.flatten()

        X_train = self._prepare_features(X, fit=True)

        # Target scaling
        y_train = y.copy()
        if self.scale_target:
            self.scaler_y = StandardScaler()
            y_train = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

        # Build network
        self.model = _ShallowNetwork(
            input_dim=X_train.shape[1],
            hidden_units=self.hidden_units,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=max(1, self.patience // 2), min_lr=1e-6
        )
        criterion = nn.MSELoss()

        # Data loaders
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            X_val_prepared = self._prepare_features(X_val, fit=False)
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            y_val_scaled = y_val.flatten()
            if self.scale_target and self.scaler_y is not None:
                y_val_scaled = self.scaler_y.transform(y_val_scaled.reshape(-1, 1)).flatten()
            X_val_tensor = torch.FloatTensor(X_val_prepared).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val_scaled).to(self.device)

        # Training loop
        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            self.history['loss'].append(avg_train_loss)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val_tensor)
                    val_loss = criterion(val_pred, y_val_tensor).item()
                self.history['val_loss'].append(val_loss)
                monitor_loss = val_loss
            else:
                monitor_loss = avg_train_loss

            scheduler.step(monitor_loss)

            if monitor_loss < best_loss:
                best_loss = monitor_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True

        # R² on original scale
        train_pred = self.predict(X)
        y_flat = y.flatten()
        ss_res = np.sum((y_flat - train_pred) ** 2)
        ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        self.training_history['r2_train'] = r2
        self.training_history['loss'] = self.history['loss']
        if self.history['val_loss']:
            self.training_history['val_loss'] = self.history['val_loss']

        self.logger.info(f"Shallow ANN fitted with R² = {r2:.4f} on {self.device} ({len(self.history['loss'])} epochs)")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        X_prepared = self._prepare_features(X, fit=False)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_prepared).to(self.device)
            preds = self.model(X_tensor).cpu().numpy()

        if self.scale_target and self.scaler_y is not None:
            preds = self.scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()

        return preds

    def get_params(self) -> dict:
        return {
            'hidden_units': self.hidden_units,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'dropout_rate': self.dropout_rate,
            'scale_inputs': self.scale_inputs,
            'scale_target': self.scale_target,
            'device': str(self.device)
        }

    def get_training_history(self) -> Optional[pd.DataFrame]:
        if not self.history['loss']:
            return None
        data = {'loss': self.history['loss']}
        if self.history['val_loss']:
            data['val_loss'] = self.history['val_loss']
        return pd.DataFrame(data)

    def summary(self) -> str:
        if self.model is None:
            return "Model not built yet"
        return str(self.model)
