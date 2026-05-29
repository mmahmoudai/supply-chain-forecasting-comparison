"""Data loading utilities for all 4 cases."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.constants import (
    CASE1_PATH, CASE2_TRAIN_PATH, CASE2_TEST_PATH, CASE3_PATH, CASE4_PATH,
    TARGET_COLUMNS, DATE_COLUMNS
)


class DataLoader:
    """Data loader for all 4 regression cases."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._datasets = {}
    
    def load_case1(self) -> pd.DataFrame:
        """
        Load Case 1: Historical Product Demand dataset.
        
        Returns:
            DataFrame with parsed dates and validated columns.
        """
        self.logger.info(f"Loading Case 1 from {CASE1_PATH}")
        
        df = pd.read_csv(CASE1_PATH)

        # Normalize BOM artifacts in column names
        df = df.rename(columns=lambda c: c.replace('\ufeff', '').replace('ï»¿', ''))
        
        # Parse date column
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
        
        # Validate target column exists
        target = TARGET_COLUMNS['case1']
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in Case 1 data")
        
        # Convert target to numeric, handling potential string values
        df[target] = pd.to_numeric(df[target].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
        
        self.logger.info(f"Case 1 loaded: {len(df)} rows, {len(df.columns)} columns")
        self._datasets['case1'] = df
        
        return df
    
    def load_case2(self, include_test: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load Case 2: Food Demand datasets (train and test).
        
        Args:
            include_test: Whether to load the test file (for predictions).
            
        Returns:
            Tuple of (train_df, test_df) or (train_df, None).
        """
        self.logger.info(f"Loading Case 2 train from {CASE2_TRAIN_PATH}")
        
        train_df = pd.read_csv(CASE2_TRAIN_PATH)
        
        # Validate target exists in train
        target = TARGET_COLUMNS['case2']
        if target not in train_df.columns:
            raise ValueError(f"Target column '{target}' not found in Case 2 train data")
        
        self.logger.info(f"Case 2 train loaded: {len(train_df)} rows")
        
        test_df = None
        if include_test:
            if CASE2_TEST_PATH.exists():
                self.logger.info(f"Loading Case 2 test from {CASE2_TEST_PATH}")
                test_df = pd.read_csv(CASE2_TEST_PATH)
                self.logger.info(f"Case 2 test loaded: {len(test_df)} rows")
            else:
                self.logger.warning(f"Case 2 test file not found at {CASE2_TEST_PATH}; continuing without test data")
        
        self._datasets['case2_train'] = train_df
        self._datasets['case2_test'] = test_df
        
        return train_df, test_df
    
    def load_case3(self) -> pd.DataFrame:
        """
        Load Case 3: Shipping Delivery Days dataset (DataCo Supply Chain).
        
        Returns:
            DataFrame with parsed dates.
        """
        self.logger.info(f"Loading Case 3 from {CASE3_PATH}")
        
        # Handle encoding issues
        try:
            df = pd.read_csv(CASE3_PATH, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(CASE3_PATH, encoding='latin-1')
        
        # Parse order date
        date_col = DATE_COLUMNS['case3']
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], format='%m/%d/%Y %H:%M', errors='coerce')
        
        # Validate target column exists
        target = TARGET_COLUMNS['case3']
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in Case 3 data")
        
        self.logger.info(f"Case 3 loaded: {len(df)} rows, {len(df.columns)} columns")
        self._datasets['case3'] = df
        
        return df
    
    def load_case4(self) -> pd.DataFrame:
        """
        Load Case 4: Retail Store Inventory dataset.
        
        Returns:
            DataFrame with parsed dates.
        """
        self.logger.info(f"Loading Case 4 from {CASE4_PATH}")
        
        df = pd.read_csv(CASE4_PATH)
        
        # Parse date column
        date_col = DATE_COLUMNS['case4']
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Validate target column exists
        target = TARGET_COLUMNS['case4']
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in Case 4 data")
        
        self.logger.info(f"Case 4 loaded: {len(df)} rows, {len(df.columns)} columns")
        self._datasets['case4'] = df
        
        return df
    
    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets.
        
        Returns:
            Dictionary with all loaded datasets.
        """
        self.logger.info("Loading all datasets...")
        
        results = {
            'case1': self.load_case1(),
            'case3': self.load_case3(),
            'case4': self.load_case4()
        }
        
        train, test = self.load_case2(include_test=True)
        results['case2_train'] = train
        results['case2_test'] = test
        
        self.logger.info("All datasets loaded successfully")
        return results
    
    def get_dataset(self, case: str) -> pd.DataFrame:
        """Get a previously loaded dataset."""
        if case not in self._datasets:
            raise ValueError(f"Dataset '{case}' not loaded. Call the appropriate load method first.")
        return self._datasets[case]
    
    def get_data_summary(self) -> pd.DataFrame:
        """Get summary statistics for all loaded datasets."""
        summaries = []
        
        for name, df in self._datasets.items():
            if df is not None:
                summary = {
                    'Dataset': name,
                    'Rows': len(df),
                    'Columns': len(df.columns),
                    'Missing Values': df.isnull().sum().sum(),
                    'Memory (MB)': df.memory_usage(deep=True).sum() / 1024 / 1024
                }
                summaries.append(summary)
        
        return pd.DataFrame(summaries)


if __name__ == "__main__":
    # Test loading
    logging.basicConfig(level=logging.INFO)
    loader = DataLoader()
    
    try:
        datasets = loader.load_all()
        print("\nData Summary:")
        print(loader.get_data_summary())
    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
