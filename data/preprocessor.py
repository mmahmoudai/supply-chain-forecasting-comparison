"""Preprocessing pipelines for all 4 cases."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import mstats
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.constants import (
    WINSOR_LOWER, WINSOR_UPPER, HIGH_CARDINALITY_THRESHOLD,
    CASE3_APPROVED_COLS, CASE3_EXCLUDED_COLS, CASE3_OPTIONAL_COLS,
    CASE3_ONEHOT_COLS, CASE3_TARGET_ENCODE_COLS, CASE3_NUMERICAL_COLS,
    TARGET_COLUMNS
)


class Preprocessor:
    """Data preprocessing for all 4 regression cases."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.scalers = {}
        self.encoders = {}
        self.target_encodings = {}
        self._fitted = False
    
    def handle_missing_values(self, df: pd.DataFrame, 
                              numerical_strategy: str = 'median',
                              categorical_strategy: str = 'Unknown') -> pd.DataFrame:
        """
        Handle missing values in the dataframe.
        
        Args:
            df: Input dataframe
            numerical_strategy: 'median', 'mean', or 'zero'
            categorical_strategy: Value to fill for categorical columns
            
        Returns:
            DataFrame with missing values handled.
        """
        df = df.copy()
        
        for col in df.columns:
            if df[col].isnull().sum() == 0:
                continue
                
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                if numerical_strategy == 'median':
                    fill_value = df[col].median()
                elif numerical_strategy == 'mean':
                    fill_value = df[col].mean()
                else:
                    fill_value = 0
                df[col] = df[col].fillna(fill_value)
            else:
                df[col] = df[col].fillna(categorical_strategy)
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame, 
                        columns: List[str],
                        method: str = 'winsorize',
                        lower_percentile: float = WINSOR_LOWER,
                        upper_percentile: float = WINSOR_UPPER) -> pd.DataFrame:
        """
        Handle outliers using winsorization or clipping.
        
        Args:
            df: Input dataframe
            columns: Columns to apply outlier treatment
            method: 'winsorize' or 'clip'
            lower_percentile: Lower percentile for winsorization
            upper_percentile: Upper percentile for winsorization
            
        Returns:
            DataFrame with outliers handled.
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if df[col].dtype not in ['float64', 'int64', 'float32', 'int32']:
                continue
            
            if method == 'winsorize':
                lower = np.percentile(df[col].dropna(), lower_percentile)
                upper = np.percentile(df[col].dropna(), upper_percentile)
                df[col] = df[col].clip(lower=lower, upper=upper)
            elif method == 'clip':
                df[col] = df[col].clip(lower=0)
        
        self.logger.info(f"Outliers handled for {len(columns)} columns using {method}")
        return df
    
    def fit_scaler(self, df: pd.DataFrame, 
                   columns: List[str], 
                   scaler_name: str = 'default') -> 'Preprocessor':
        """
        Fit a StandardScaler on specified columns.
        
        Args:
            df: Training dataframe
            columns: Columns to scale
            scaler_name: Name identifier for the scaler
            
        Returns:
            self for chaining
        """
        valid_cols = [c for c in columns if c in df.columns and 
                      df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        if not valid_cols:
            return self
        
        scaler = StandardScaler()
        scaler.fit(df[valid_cols])
        self.scalers[scaler_name] = {'scaler': scaler, 'columns': valid_cols}
        
        self.logger.info(f"Fitted scaler '{scaler_name}' on {len(valid_cols)} columns")
        return self
    
    def transform_scaler(self, df: pd.DataFrame, 
                         scaler_name: str = 'default') -> pd.DataFrame:
        """
        Apply a fitted scaler to the dataframe.
        
        Args:
            df: Dataframe to transform
            scaler_name: Name of the fitted scaler
            
        Returns:
            Transformed dataframe.
        """
        if scaler_name not in self.scalers:
            raise ValueError(f"Scaler '{scaler_name}' not fitted")
        
        df = df.copy()
        scaler_info = self.scalers[scaler_name]
        scaler = scaler_info['scaler']
        columns = scaler_info['columns']
        
        valid_cols = [c for c in columns if c in df.columns]
        if valid_cols:
            df[valid_cols] = scaler.transform(df[valid_cols])
        
        return df
    
    def fit_target_encoding(self, df: pd.DataFrame, 
                            columns: List[str],
                            target_col: str,
                            encoding_name: str = 'default') -> 'Preprocessor':
        """
        Fit target encoding for high-cardinality categorical columns.
        
        Args:
            df: Training dataframe
            columns: Columns to encode
            target_col: Target column for encoding
            encoding_name: Name identifier for the encoding
            
        Returns:
            self for chaining
        """
        encodings = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Calculate mean target per category
            mean_target = df.groupby(col)[target_col].mean().to_dict()
            global_mean = df[target_col].mean()
            
            encodings[col] = {
                'mapping': mean_target,
                'global_mean': global_mean
            }
        
        self.target_encodings[encoding_name] = encodings
        self.logger.info(f"Fitted target encoding for {len(columns)} columns")
        return self
    
    def transform_target_encoding(self, df: pd.DataFrame,
                                   encoding_name: str = 'default') -> pd.DataFrame:
        """
        Apply target encoding to the dataframe.
        
        Args:
            df: Dataframe to transform
            encoding_name: Name of the fitted encoding
            
        Returns:
            Transformed dataframe with encoded columns.
        """
        if encoding_name not in self.target_encodings:
            raise ValueError(f"Target encoding '{encoding_name}' not fitted")
        
        df = df.copy()
        encodings = self.target_encodings[encoding_name]
        
        for col, enc_info in encodings.items():
            if col not in df.columns:
                continue
            
            new_col_name = f"{col}_encoded"
            df[new_col_name] = df[col].map(enc_info['mapping'])
            df[new_col_name] = df[new_col_name].fillna(enc_info['global_mean'])
            df = df.drop(columns=[col])
        
        return df
    
    def one_hot_encode(self, df: pd.DataFrame, 
                       columns: List[str],
                       drop_first: bool = True) -> pd.DataFrame:
        """
        Apply one-hot encoding to categorical columns.
        
        Args:
            df: Dataframe to encode
            columns: Columns to one-hot encode
            drop_first: Whether to drop first category to avoid multicollinearity
            
        Returns:
            DataFrame with one-hot encoded columns.
        """
        valid_cols = [c for c in columns if c in df.columns]
        
        if not valid_cols:
            return df
        
        df = pd.get_dummies(df, columns=valid_cols, drop_first=drop_first)
        self.logger.info(f"One-hot encoded {len(valid_cols)} columns")
        
        return df
    
    def preprocess_case1(self, df: pd.DataFrame, 
                         is_training: bool = True) -> pd.DataFrame:
        """
        Preprocess Case 1: Historical Product Demand.
        
        Args:
            df: Raw dataframe
            is_training: Whether this is training data
            
        Returns:
            Preprocessed dataframe.
        """
        df = df.copy()
        target = TARGET_COLUMNS['case1']
        
        # Handle negative demand (clip to 0)
        df[target] = df[target].clip(lower=0)
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Handle outliers in target
        if is_training:
            df = self.handle_outliers(df, [target])
        
        self.logger.info(f"Case 1 preprocessed: {len(df)} rows")
        return df
    
    def preprocess_case2(self, df: pd.DataFrame, 
                         is_training: bool = True) -> pd.DataFrame:
        """
        Preprocess Case 2: Food Demand.
        
        Args:
            df: Raw dataframe
            is_training: Whether this is training data
            
        Returns:
            Preprocessed dataframe.
        """
        df = df.copy()
        target = TARGET_COLUMNS['case2']
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        self.logger.info(f"Case 2 preprocessed: {len(df)} rows")
        return df
    
    def preprocess_case3(self, df: pd.DataFrame,
                         is_training: bool = True) -> pd.DataFrame:
        """
        Preprocess Case 3: Shipping Delivery Days (with leakage prevention).

        Follows exact preprocessing rules from PDF:
        - Missing values: median for numerical, "Unknown" for categorical
        - Outlier treatment: Winsorization at 1st-99th percentiles
        - One-hot encoding for low-cardinality categoricals
        - Target encoding for high-cardinality columns
        - Standardization for numerical features

        Args:
            df: Raw dataframe
            is_training: Whether this is training data

        Returns:
            Preprocessed dataframe with only approved columns.
        """
        df = df.copy()
        target = TARGET_COLUMNS['case3']

        # CRITICAL: Select only approved columns (leakage prevention)
        approved_cols = [target] + [c for c in CASE3_APPROVED_COLS if c in df.columns]
        # Optionally include additional approved columns
        for opt_col in CASE3_OPTIONAL_COLS:
            if opt_col in df.columns and opt_col not in approved_cols:
                approved_cols.append(opt_col)

        df = df[approved_cols]
        self.logger.info(f"Case 3: Selected {len(approved_cols)} approved columns (leakage prevention)")

        # 2.1 Handle missing values per PDF rules
        # Numerical: median imputation
        # Categorical: replace with "Unknown"
        df = self.handle_missing_values(df, numerical_strategy='median',
                                        categorical_strategy='Unknown')

        # 2.2 Handle outliers using winsorization at 1st and 99th percentiles
        # Apply to continuous variables as specified in PDF
        existing_numerical = [c for c in CASE3_NUMERICAL_COLS if c in df.columns]
        if is_training and existing_numerical:
            df = self.handle_outliers(df, existing_numerical, method='winsorize',
                                     lower_percentile=WINSOR_LOWER,
                                     upper_percentile=WINSOR_UPPER)

        # 3.1 Target encoding for high-cardinality columns (Order Country, Order State)
        # Must compute means using training set only
        existing_target_enc = [c for c in CASE3_TARGET_ENCODE_COLS if c in df.columns]
        if is_training and existing_target_enc:
            self.fit_target_encoding(df, existing_target_enc, target, 'case3_target_enc')
            df = self.transform_target_encoding(df, 'case3_target_enc')
        elif 'case3_target_enc' in self.target_encodings:
            df = self.transform_target_encoding(df, 'case3_target_enc')

        self.logger.info(f"Case 3 preprocessed: {len(df)} rows")
        return df
    
    def preprocess_case4(self, df: pd.DataFrame, 
                         is_training: bool = True) -> pd.DataFrame:
        """
        Preprocess Case 4: Retail Store Inventory.
        
        Args:
            df: Raw dataframe
            is_training: Whether this is training data
            
        Returns:
            Preprocessed dataframe.
        """
        df = df.copy()
        target = TARGET_COLUMNS['case4']
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Handle outliers in target
        if is_training:
            df = self.handle_outliers(df, [target])
        
        self.logger.info(f"Case 4 preprocessed: {len(df)} rows")
        return df
