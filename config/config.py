"""Configuration class for the 4-Case Regression Analysis project."""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

from .constants import (
    RANDOM_SEED, TRAIN_RATIO, FIGURE_DPI, FIGURE_STYLE,
    XGBOOST_DEFAULTS, RANDOM_FOREST_DEFAULTS, FFNN_DEFAULTS,
    OUTPUTS_DIR, FIGURES_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR
)


@dataclass
class Config:
    """Configuration class for experiment settings."""
    
    # General settings
    random_seed: int = RANDOM_SEED
    train_ratio: float = TRAIN_RATIO
    verbose: bool = True
    
    # Figure settings
    figure_dpi: int = FIGURE_DPI
    figure_style: str = FIGURE_STYLE
    save_figures: bool = True
    
    # Model settings
    xgboost_params: Dict = field(default_factory=lambda: XGBOOST_DEFAULTS.copy())
    random_forest_params: Dict = field(default_factory=lambda: RANDOM_FOREST_DEFAULTS.copy())
    ffnn_params: Dict = field(default_factory=lambda: FFNN_DEFAULTS.copy())
    
    # Paths
    output_dir: Path = OUTPUTS_DIR
    figures_dir: Path = FIGURES_DIR
    models_dir: Path = MODELS_DIR
    results_dir: Path = RESULTS_DIR
    logs_dir: Path = LOGS_DIR
    
    # Experiment flags
    run_classical_models: bool = True
    run_ml_models: bool = True
    run_neural_networks: bool = True
    
    # Cases to run
    cases_to_run: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    
    def __post_init__(self):
        """Create output directories if they don't exist."""
        self._create_directories()
        self._setup_logging()
    
    def _create_directories(self):
        """Create all necessary output directories."""
        dirs = [
            self.output_dir,
            self.figures_dir,
            self.models_dir,
            self.results_dir,
            self.logs_dir,
            self.figures_dir / "chapter6",
            self.figures_dir / "chapter7",
            self.models_dir / "classical",
            self.models_dir / "ml"
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.logs_dir / "experiment.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name."""
        return logging.getLogger(name)
    
    def update_params(self, model_type: str, **kwargs):
        """Update model parameters."""
        if model_type == 'xgboost':
            self.xgboost_params.update(kwargs)
        elif model_type == 'random_forest':
            self.random_forest_params.update(kwargs)
        elif model_type == 'ffnn':
            self.ffnn_params.update(kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'random_seed': self.random_seed,
            'train_ratio': self.train_ratio,
            'figure_dpi': self.figure_dpi,
            'xgboost_params': self.xgboost_params,
            'random_forest_params': self.random_forest_params,
            'ffnn_params': self.ffnn_params,
            'cases_to_run': self.cases_to_run
        }
