"""Helper utilities for the 4-Case Regression Analysis project."""

import numpy as np
import pandas as pd
import random
import time
from pathlib import Path
from typing import Union, Callable
from functools import wraps
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.constants import RANDOM_SEED


def set_random_seeds(seed: int = RANDOM_SEED):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def timer(func: Callable) -> Callable:
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__name__)
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"{func.__name__} completed in {duration:.2f} seconds")
        return result
    return wrapper


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame, 
                   filepath: Union[str, Path],
                   format: str = 'csv') -> Path:
    """
    Save a DataFrame to disk.
    
    Args:
        df: DataFrame to save
        filepath: Output path
        format: 'csv', 'parquet', or 'pickle'
        
    Returns:
        Path to saved file
    """
    filepath = Path(filepath)
    ensure_directory(filepath.parent)
    
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'parquet':
        df.to_parquet(filepath, index=False)
    elif format == 'pickle':
        df.to_pickle(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return filepath


def load_dataframe(filepath: Union[str, Path],
                   format: str = None) -> pd.DataFrame:
    """
    Load a DataFrame from disk.
    
    Args:
        filepath: Input path
        format: 'csv', 'parquet', 'pickle', or None (auto-detect)
        
    Returns:
        Loaded DataFrame
    """
    filepath = Path(filepath)
    
    if format is None:
        # Auto-detect based on extension
        ext = filepath.suffix.lower()
        format_map = {
            '.csv': 'csv',
            '.parquet': 'parquet',
            '.pkl': 'pickle',
            '.pickle': 'pickle'
        }
        format = format_map.get(ext, 'csv')
    
    if format == 'csv':
        return pd.read_csv(filepath)
    elif format == 'parquet':
        return pd.read_parquet(filepath)
    elif format == 'pickle':
        return pd.read_pickle(filepath)
    else:
        raise ValueError(f"Unknown format: {format}")


def get_memory_usage(df: pd.DataFrame) -> str:
    """
    Get memory usage of a DataFrame in human-readable format.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Memory usage string
    """
    bytes_used = df.memory_usage(deep=True).sum()
    
    if bytes_used < 1024:
        return f"{bytes_used} B"
    elif bytes_used < 1024**2:
        return f"{bytes_used/1024:.2f} KB"
    elif bytes_used < 1024**3:
        return f"{bytes_used/1024**2:.2f} MB"
    else:
        return f"{bytes_used/1024**3:.2f} GB"


def reduce_memory_usage(df: pd.DataFrame, 
                        verbose: bool = True) -> pd.DataFrame:
    """
    Reduce memory usage of a DataFrame by optimizing dtypes.
    
    Args:
        df: DataFrame to optimize
        verbose: Whether to print memory reduction info
        
    Returns:
        Optimized DataFrame
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                    
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose:
        logger = logging.getLogger('reduce_memory')
        logger.info(f"Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
                   f"({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    
    return df


class ProgressTracker:
    """Simple progress tracker for long-running operations."""
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        self.logger = logging.getLogger('ProgressTracker')
    
    def update(self, n: int = 1):
        """Update progress by n steps."""
        self.current += n
        elapsed = time.time() - self.start_time
        
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
        else:
            eta = 0
        
        pct = 100 * self.current / self.total
        self.logger.info(f"{self.description}: {pct:.1f}% ({self.current}/{self.total}) "
                        f"- ETA: {eta:.1f}s")
    
    def finish(self):
        """Mark progress as complete."""
        elapsed = time.time() - self.start_time
        self.logger.info(f"{self.description}: Complete in {elapsed:.2f}s")
