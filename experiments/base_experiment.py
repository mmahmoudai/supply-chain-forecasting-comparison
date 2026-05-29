"""Base experiment class for all cases."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from data.data_loader import DataLoader
from data.preprocessor import Preprocessor
from data.feature_engineer import FeatureEngineer
from evaluation.metrics import MetricsCalculator
from evaluation.visualizations import Visualizer


class BaseExperiment(ABC):
    """Abstract base class for case experiments."""
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the experiment.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.data_loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.feature_engineer = FeatureEngineer()
        self.metrics = MetricsCalculator()
        self.visualizer = Visualizer()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Model storage
        self.models = {}
        self.predictions = {}
        self.results = {}
    
    @property
    @abstractmethod
    def case_name(self) -> str:
        """Return the case identifier."""
        pass
    
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        """Load the raw dataset for this case."""
        pass
    
    @abstractmethod
    def preprocess(self) -> pd.DataFrame:
        """Preprocess the data."""
        pass
    
    @abstractmethod
    def engineer_features(self) -> pd.DataFrame:
        """Create features for the models."""
        pass
    
    @abstractmethod
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        pass
    
    @abstractmethod
    def train_classical_models(self) -> Dict:
        """Train classical statistical models."""
        pass
    
    @abstractmethod
    def train_ml_models(self) -> Dict:
        """Train machine learning models."""
        pass
    
    def evaluate_models(self) -> pd.DataFrame:
        """
        Evaluate all trained models.
        
        Returns:
            DataFrame with evaluation results
        """
        results_list = []
        
        for model_name, model in self.models.items():
            try:
                # Get predictions
                if hasattr(model, 'forecast'):
                    # Time series model
                    preds = model.forecast(len(self.y_test))
                else:
                    # Supervised model
                    preds = model.predict(self.X_test)
                
                self.predictions[model_name] = preds
                
                # Calculate metrics
                metrics = self.metrics.evaluate_model(
                    model_name=model_name,
                    y_true=self.y_test,
                    y_pred=preds,
                    case=self.case_name,
                    dataset_split='test'
                )
                
                results_list.append({
                    'model': model_name,
                    **metrics
                })
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
        
        self.results = pd.DataFrame(results_list)
        return self.results
    
    def generate_figures(self):
        """Generate visualization figures for this case."""
        self.logger.info(f"Generating figures for {self.case_name}")
    
    def save_results(self):
        """Save experiment results to disk."""
        results_dir = self.config.results_dir
        
        # Save metrics
        if not self.results.empty if isinstance(self.results, pd.DataFrame) else self.results:
            results_path = results_dir / f"{self.case_name}_results.csv"
            if isinstance(self.results, pd.DataFrame):
                self.results.to_csv(results_path, index=False)
            self.logger.info(f"Results saved to {results_path}")
        
        # Save predictions
        for model_name, preds in self.predictions.items():
            pred_path = results_dir / f"{self.case_name}_{model_name}_predictions.csv"
            pd.DataFrame({'prediction': preds}).to_csv(pred_path, index=False)

        # Save model configurations and config figures
        self.save_model_configs()
    
    def save_models(self):
        """Save trained models to disk."""
        for model_name, model in self.models.items():
            try:
                model_path = self.config.models_dir / f"{self.case_name}_{model_name}.pkl"
                model.save(model_path)
            except Exception as e:
                self.logger.warning(f"Could not save {model_name}: {e}")

    def _stringify_param(self, value: Any) -> str:
        """Convert parameter values into stable string representations."""
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(v) for v in value)
        if isinstance(value, dict):
            try:
                return json.dumps(value, sort_keys=True)
            except TypeError:
                return str(value)
        if isinstance(value, np.ndarray):
            return np.array2string(value, precision=6, separator=",")
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def save_model_configs(self):
        """Save model configuration artifacts (JSON/CSV + figure)."""
        config_dir = self.config.results_dir / "model_configs"
        config_dir.mkdir(parents=True, exist_ok=True)

        feature_count = None
        if hasattr(self, 'feature_cols') and self.feature_cols is not None:
            feature_count = len(self.feature_cols)

        for model_name, model in self.models.items():
            params = {}
            if hasattr(model, 'get_params'):
                try:
                    params = model.get_params() or {}
                except Exception as e:
                    self.logger.warning(f"Failed to read params for {model_name}: {e}")
                    params = {}

            config_payload = {
                'case': self.case_name,
                'model': model_name
            }
            if feature_count is not None:
                config_payload['feature_count'] = feature_count

            for key, value in params.items():
                config_payload[key] = self._stringify_param(value)

            json_path = config_dir / f"{self.case_name}_{model_name}_config.json"
            with open(json_path, 'w') as f:
                json.dump(config_payload, f, indent=2)

            csv_path = config_dir / f"{self.case_name}_{model_name}_config.csv"
            df = pd.DataFrame({
                'Parameter': list(config_payload.keys()),
                'Value': [config_payload[k] for k in config_payload.keys()]
            })
            df.to_csv(csv_path, index=False)

            title = f"Model Config: {self.case_name} - {model_name}"
            filename = f"config_{self.case_name}_{model_name}"
            self.visualizer.plot_model_config_table(
                params={k: str(v) for k, v in config_payload.items()},
                title=title,
                filename=filename,
                chapter='chapter7'
            )
    
    def run(self, save_results: bool = True) -> pd.DataFrame:
        """
        Run the complete experiment pipeline.
        
        Args:
            save_results: Whether to save results to disk
            
        Returns:
            DataFrame with evaluation results
        """
        self.logger.info(f"Starting experiment: {self.case_name}")
        
        # Data pipeline
        self.logger.info("Loading data...")
        self.load_data()
        
        self.logger.info("Preprocessing data...")
        self.preprocess()
        
        self.logger.info("Engineering features...")
        self.engineer_features()
        
        self.logger.info("Splitting data...")
        self.split_data()
        
        # Model training
        if self.config.run_classical_models:
            self.logger.info("Training classical models...")
            self.train_classical_models()
        
        if self.config.run_ml_models:
            self.logger.info("Training ML models...")
            self.train_ml_models()
        
        # Evaluation
        self.logger.info("Evaluating models...")
        results = self.evaluate_models()
        
        # Generate figures
        self.logger.info("Generating figures...")
        self.generate_figures()
        
        # Save results
        if save_results:
            self.save_results()
            self.save_models()
        
        self.logger.info(f"Experiment {self.case_name} completed!")
        return results
    
    def get_best_model(self, metric: str = 'RMSE') -> Tuple[str, Any]:
        """
        Get the best performing model.

        Args:
            metric: Metric to use for comparison

        Returns:
            Tuple of (model_name, model_object)
        """
        best_model_name = self.metrics.get_best_model(
            case=self.case_name,
            metric=metric
        )

        if best_model_name and best_model_name in self.models:
            return best_model_name, self.models[best_model_name]

        return None, None

    def run_accuracy_vs_size_analysis(self,
                                       model_classes: Dict[str, type],
                                       sizes: List[float] = [0.25, 0.5, 0.75, 1.0],
                                       metric: str = 'RMSE') -> Dict[str, List[float]]:
        """
        Analyze model performance across different training data sizes (Figure 7.13).

        Args:
            model_classes: Dict of {model_name: ModelClass} to evaluate
            sizes: Fractions of training data to use
            metric: Metric to track (RMSE or R2)

        Returns:
            Dict of {model_name: [scores_at_each_size]}
        """
        self.logger.info(f"Running accuracy vs dataset size analysis for {self.case_name}")

        results = {name: [] for name in model_classes.keys()}
        n_train = len(self.X_train)

        for size in sizes:
            n_samples = int(n_train * size)
            X_subset = self.X_train.iloc[:n_samples] if isinstance(self.X_train, pd.DataFrame) else self.X_train[:n_samples]
            y_subset = self.y_train[:n_samples]

            for model_name, model_class in model_classes.items():
                try:
                    # Create and train model on subset
                    model = model_class()
                    model.fit(X_subset, y_subset)

                    # Predict on test set
                    preds = model.predict(self.X_test)

                    # Calculate metric
                    if metric == 'RMSE':
                        score = np.sqrt(np.mean((self.y_test - preds) ** 2))
                    elif metric == 'R2':
                        ss_res = np.sum((self.y_test - preds) ** 2)
                        ss_tot = np.sum((self.y_test - np.mean(self.y_test)) ** 2)
                        score = 1 - (ss_res / ss_tot)
                    else:
                        score = np.mean(np.abs(self.y_test - preds))  # MAE

                    results[model_name].append(score)
                    self.logger.info(f"  {model_name} at {int(size*100)}%: {metric}={score:.4f}")

                except Exception as e:
                    self.logger.error(f"Error training {model_name} at {size}: {e}")
                    results[model_name].append(np.nan)

        return results

    def generate_accuracy_vs_size_figure(self,
                                          model_classes: Dict[str, type],
                                          sizes: List[float] = [0.25, 0.5, 0.75, 1.0],
                                          metric: str = 'RMSE'):
        """
        Generate accuracy vs dataset size figure (Figure 7.13).

        Args:
            model_classes: Dict of {model_name: ModelClass} to evaluate
            sizes: Fractions of training data to use
            metric: Metric to track
        """
        scores = self.run_accuracy_vs_size_analysis(model_classes, sizes, metric)

        self.visualizer.plot_accuracy_vs_dataset_size(
            sizes=sizes,
            model_scores=scores,
            metric_name=metric,
            title=f'Model Performance vs Training Data Size ({self.case_name})',
            filename=f'figure_7_13_{self.case_name}_accuracy_vs_size',
            chapter='chapter7'
        )
