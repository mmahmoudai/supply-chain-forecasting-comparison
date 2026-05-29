"""Feedforward Neural Network (FFNN) model using PyTorch with GPU support."""

import numpy as np
import pandas as pd
from typing import Union, Optional, List
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from models.base import SupervisedModel
from config.constants import RANDOM_SEED, FFNN_DEFAULTS

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _get_device():
    """Get the best available device (CUDA GPU or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


class _FFNNNetwork(nn.Module):
    """PyTorch neural network architecture for FFNN."""

    def __init__(self, input_dim: int, hidden_layers: List[int],
                 activation: str = 'relu', dropout_rate: float = 0.2):
        super().__init__()

        layers = []
        # Input batch normalization
        layers.append(nn.BatchNorm1d(input_dim))

        prev_dim = input_dim
        for i, units in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, units))
            layers.append(nn.BatchNorm1d(units))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = units

        # Output layer (linear for regression)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class FFNNModel(SupervisedModel):
    """
    Feedforward Neural Network (FFNN) for regression using PyTorch.

    Uses GPU acceleration when available (CUDA).
    """

    def __init__(self,
                 hidden_layers: List[int] = None,
                 activation: str = 'relu',
                 learning_rate: float = 0.001,
                 batch_size: int = 64,
                 epochs: int = 200,
                 patience: int = 20,
                 dropout_rate: float = 0.2,
                 **kwargs):
        """
        Initialize the FFNN model.

        Args:
            hidden_layers: List of hidden layer sizes [128, 64, 32]
            activation: Activation function for hidden layers
            learning_rate: Learning rate for Adam optimizer
            batch_size: Training batch size
            epochs: Maximum training epochs
            patience: Early stopping patience
            dropout_rate: Dropout rate between layers
        """
        super().__init__("FFNN")

        self.device = _get_device()

        # Set seeds
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(RANDOM_SEED)

        # Model architecture
        self.hidden_layers = hidden_layers or FFNN_DEFAULTS['hidden_layers']
        self.activation = activation
        self.dropout_rate = dropout_rate

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience

        self.history = {'loss': [], 'val_loss': []}

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray],
            X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
            y_val: Optional[Union[pd.Series, np.ndarray]] = None,
            verbose: int = 0,
            **kwargs) -> 'FFNNModel':
        """
        Fit the FFNN model.

        Args:
            X: Feature matrix
            y: Target values
            X_val: Validation features
            y_val: Validation targets
            verbose: Verbosity level

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
        y = y.flatten()

        # Build network
        self.model = _FFNNNetwork(
            input_dim=X.shape[1],
            hidden_layers=self.hidden_layers,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=self.patience // 2, min_lr=1e-6
        )
        criterion = nn.MSELoss()

        # Prepare data loaders
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        has_val = X_val is not None and y_val is not None
        if has_val:
            if isinstance(X_val, pd.DataFrame):
                X_val = X_val.values
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            y_val = y_val.flatten()
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val).to(self.device)

        # Training loop with early stopping
        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Train
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

            # Validation
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

            # Early stopping
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

        # Restore best weights
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.is_fitted = True

        # Calculate training R²
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_tensor).cpu().numpy()
        ss_res = np.sum((y - train_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        self.training_history['r2_train'] = r2
        self.training_history['loss'] = self.history['loss']
        if self.history['val_loss']:
            self.training_history['val_loss'] = self.history['val_loss']

        self.logger.info(f"FFNN fitted with R² = {r2:.4f} on {self.device} ({len(self.history['loss'])} epochs)")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray], **kwargs) -> np.ndarray:
        """Generate predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if isinstance(X, pd.DataFrame):
            X = X.values

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()

        return predictions

    def get_params(self) -> dict:
        """Get model parameters."""
        return {
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'dropout_rate': self.dropout_rate,
            'device': str(self.device)
        }

    def get_training_history(self) -> pd.DataFrame:
        """Get training history as DataFrame."""
        if not self.history['loss']:
            return None
        data = {'loss': self.history['loss']}
        if self.history['val_loss']:
            # Pad if lengths differ
            data['val_loss'] = self.history['val_loss']
        return pd.DataFrame(data)

    def summary(self) -> str:
        """Get model architecture summary."""
        if self.model is None:
            return "Model not built yet"
        return str(self.model)

    def save_model(self, filepath: Union[str, Path]):
        """Save PyTorch model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'params': self.get_params(),
            'feature_names': self.feature_names
        }, str(filepath) + '.pt')
        self.logger.info(f"Model saved to {filepath}.pt")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> 'FFNNModel':
        """Load a saved PyTorch model."""
        filepath = Path(filepath)
        checkpoint = torch.load(str(filepath), map_location=_get_device())
        params = checkpoint['params']
        instance = cls(
            hidden_layers=params.get('hidden_layers'),
            activation=params.get('activation', 'relu'),
            dropout_rate=params.get('dropout_rate', 0.2)
        )
        instance.feature_names = checkpoint.get('feature_names')
        n_features = len(instance.feature_names) if instance.feature_names else 1
        instance.model = _FFNNNetwork(
            input_dim=n_features,
            hidden_layers=instance.hidden_layers,
            activation=instance.activation,
            dropout_rate=instance.dropout_rate
        ).to(instance.device)
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.is_fitted = True
        return instance
