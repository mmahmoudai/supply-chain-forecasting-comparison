"""Feature engineering for all 4 cases."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from config.constants import (
    LAG_FEATURES, ROLLING_WINDOWS, TARGET_COLUMNS, DATE_COLUMNS,
    CASE2_TRAIN_WEEKS, CASE2_VAL_WEEKS, CASE2_TEST_WEEKS,
    CASE3_ONEHOT_COLS, CASE3_TARGET_ENCODE_COLS
)


class FeatureEngineer:
    """Feature engineering for all 4 regression cases."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.feature_names = {}
        self.cat_cols = {}  # Store categorical column names per case
    
    def create_lag_features(self, df: pd.DataFrame, 
                            target_col: str,
                            lags: List[int],
                            group_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create lag features for the target column.
        
        Args:
            df: Input dataframe (must be sorted by time)
            target_col: Column to create lags for
            lags: List of lag values [1, 7, 14]
            group_cols: Columns to group by for panel data
            
        Returns:
            DataFrame with lag features added.
        """
        df = df.copy()
        
        for lag in lags:
            col_name = f'lag_{lag}'
            if group_cols:
                df[col_name] = df.groupby(group_cols)[target_col].shift(lag)
            else:
                df[col_name] = df[target_col].shift(lag)
        
        self.logger.info(f"Created {len(lags)} lag features")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                                 target_col: str,
                                 windows: List[int],
                                 group_cols: Optional[List[str]] = None,
                                 stats: List[str] = ['mean', 'std']) -> pd.DataFrame:
        """
        Create rolling statistics features.
        
        Args:
            df: Input dataframe (must be sorted by time)
            target_col: Column to compute rolling stats on
            windows: List of window sizes [7, 14]
            group_cols: Columns to group by for panel data
            stats: Statistics to compute ['mean', 'std']
            
        Returns:
            DataFrame with rolling features added.
        """
        df = df.copy()
        
        for window in windows:
            for stat in stats:
                col_name = f'rolling_{stat}_{window}'
                
                if group_cols:
                    if stat == 'mean':
                        df[col_name] = df.groupby(group_cols)[target_col].transform(
                            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
                        )
                    elif stat == 'std':
                        df[col_name] = df.groupby(group_cols)[target_col].transform(
                            lambda x: x.shift(1).rolling(window, min_periods=1).std()
                        )
                else:
                    if stat == 'mean':
                        df[col_name] = df[target_col].shift(1).rolling(window, min_periods=1).mean()
                    elif stat == 'std':
                        df[col_name] = df[target_col].shift(1).rolling(window, min_periods=1).std()
        
        self.logger.info(f"Created rolling features for {len(windows)} windows")
        return df
    
    def create_calendar_features(self, df: pd.DataFrame,
                                  date_col: str) -> pd.DataFrame:
        """
        Create calendar-based features from a date column.
        
        Args:
            df: Input dataframe
            date_col: Name of the date column
            
        Returns:
            DataFrame with calendar features added.
        """
        df = df.copy()
        
        if date_col not in df.columns:
            self.logger.warning(f"Date column '{date_col}' not found")
            return df
        
        # Ensure datetime type
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract features
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
        week = df[date_col].dt.isocalendar().week
        df['week_of_year'] = week.fillna(0).astype(int)
        df['quarter'] = df[date_col].dt.quarter
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        
        self.logger.info("Created calendar features")
        return df

    def create_cyclical_features(self, df: pd.DataFrame,
                                  col: str,
                                  period: int,
                                  prefix: Optional[str] = None) -> pd.DataFrame:
        """
        Create sine/cosine cyclical features for a periodic column.

        Args:
            df: Input dataframe
            col: Column name to transform
            period: Period of the cycle (e.g., 12 for months)
            prefix: Optional prefix for new columns

        Returns:
            DataFrame with cyclical features added.
        """
        if col not in df.columns:
            return df

        df = df.copy()
        name = prefix or col
        values = pd.to_numeric(df[col], errors='coerce').fillna(0)

        df[f'{name}_sin'] = np.sin(2 * np.pi * values / period)
        df[f'{name}_cos'] = np.cos(2 * np.pi * values / period)
        return df
    
    def engineer_case1_features(self, df: pd.DataFrame,
                                 aggregate: bool = True) -> pd.DataFrame:
        """
        Feature engineering for Case 1: Historical Product Demand.
        
        Args:
            df: Preprocessed dataframe
            aggregate: Whether to aggregate by date (for classical models)
            
        Returns:
            Feature-engineered dataframe.
        """
        target = TARGET_COLUMNS['case1']
        date_col = DATE_COLUMNS['case1']
        
        if aggregate:
            # Aggregate by date for univariate models
            df_agg = df.groupby(date_col)[target].sum().reset_index()
            df_agg = df_agg.sort_values(date_col)
            df = df_agg
        else:
            df = df.sort_values(date_col)
        
        # Create lag features
        lags = LAG_FEATURES['case1']
        df = self.create_lag_features(df, target, lags)
        
        # Create rolling features
        windows = ROLLING_WINDOWS['case1']
        df = self.create_rolling_features(df, target, windows)
        
        # Create calendar features
        df = self.create_calendar_features(df, date_col)

        # Time index and cyclical features
        if date_col in df.columns:
            df['time_index'] = (df[date_col] - df[date_col].min()).dt.days
        df = self.create_cyclical_features(df, 'month', 12, 'month')
        df = self.create_cyclical_features(df, 'week_of_year', 52, 'week_of_year')
        df = self.create_cyclical_features(df, 'day_of_week', 7, 'day_of_week')
        df = self.create_cyclical_features(df, 'quarter', 4, 'quarter')
        
        # Drop NaN rows from lag creation
        df = df.dropna()
        
        self.feature_names['case1'] = [c for c in df.columns 
                                        if c not in [target, date_col]]
        
        self.logger.info(f"Case 1 features: {len(self.feature_names['case1'])} features")
        return df
    
    def engineer_case2_features(self, train_df: pd.DataFrame,
                                 test_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Feature engineering for Case 2: Food Demand.
        
        Args:
            train_df: Training dataframe (weeks 1-145)
            test_df: Test dataframe for predictions (weeks 146-155, no target)
            
        Returns:
            Tuple of (train_processed, test_processed or None).
        """
        target = TARGET_COLUMNS['case2']
        
        # Combine for consistent feature engineering
        if test_df is not None:
            test_df = test_df.copy()
            test_df[target] = np.nan  # Placeholder
            df = pd.concat([train_df, test_df], ignore_index=True)
            is_combined = True
        else:
            df = train_df.copy()
            is_combined = False
        
        # Determine groupby keys based on available columns (v3 has city_code, not center_id)
        if 'center_id' in df.columns and 'meal_id' in df.columns:
            group_keys = ['center_id', 'meal_id']
            id_cols = ['center_id', 'meal_id']
        else:
            group_keys = ['city_code', 'category']
            id_cols = ['city_code']
        
        df = df.sort_values(group_keys + ['week'])
        
        # 1. Price ratio and log features
        df['price_ratio'] = df['checkout_price'] / df['base_price'].replace(0, np.nan)
        df['price_ratio'] = df['price_ratio'].fillna(0)
        df['log_checkout_price'] = np.log1p(df['checkout_price'].clip(lower=0))
        df['log_base_price'] = np.log1p(df['base_price'].clip(lower=0))

        # 2. Discount features
        df['discount_amount'] = df['base_price'] - df['checkout_price']
        df['discount_percent'] = df['discount_amount'] / df['base_price'].replace(0, 1) * 100
        df['discount_yn'] = (df['discount_amount'] > 0).astype(int)
        
        # 3. Price change features (week-over-week)
        df['prev_checkout_price'] = df.groupby(group_keys)['checkout_price'].shift(1)
        df['price_change'] = df['checkout_price'] - df['prev_checkout_price']
        df['price_change_yn'] = (df['price_change'] > 0).astype(int)
        df['price_change_pct'] = df['price_change'] / df['prev_checkout_price'].replace(0, np.nan)
        df['price_change_pct'] = df['price_change_pct'].fillna(0)
        
        # 4. Lag features (within center-meal combination)
        lags = LAG_FEATURES['case2']
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby(group_keys)[target].shift(lag)
        
        # 5. Rolling statistics
        windows = ROLLING_WINDOWS['case2']
        for window in windows:
            df[f'rolling_mean_{window}'] = df.groupby(group_keys)[target].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby(group_keys)[target].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )
        
        # 6. Time features from week number
        df['quarter'] = ((df['week'] - 1) // 13) + 1
        df['year'] = ((df['week'] - 1) // 52) + 1
        df['week_in_year'] = ((df['week'] - 1) % 52) + 1

        df = self.create_cyclical_features(df, 'week_in_year', 52, 'week_in_year')
        df = self.create_cyclical_features(df, 'quarter', 4, 'quarter')
        
        # 7. Promotion interaction
        df['promo_count'] = df['emailer_for_promotion'] + df['homepage_featured']
        df['both_promo'] = df['emailer_for_promotion'] * df['homepage_featured']
        df['promo_discount_interaction'] = df['promo_count'] * df['discount_percent']
        
        # 8. Store categorical columns (encoding deferred to encode_for_model)
        categorical_cols = ['Region', 'center_type', 'category', 'cuisine']
        self.cat_cols['case2'] = [c for c in categorical_cols if c in df.columns]
        
        # 9. Log transformation of target (train only)
        df['log_num_orders'] = np.log1p(df[target])
        
        # Drop unnecessary columns
        cols_to_drop = ['id', 'prev_checkout_price']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        # Split back if combined
        if is_combined:
            train_processed = df[df['week'] <= 145].copy()
            test_processed = df[df['week'] > 145].copy()
            
            # Store feature names
            exclude = [target, 'log_num_orders', 'week'] + id_cols
            self.feature_names['case2'] = [c for c in train_processed.columns 
                                            if c not in exclude]
            
            self.logger.info(f"Case 2 features: {len(self.feature_names['case2'])} features")
            return train_processed, test_processed
        else:
            exclude = [target, 'log_num_orders', 'week'] + id_cols
            self.feature_names['case2'] = [c for c in df.columns 
                                            if c not in exclude]
            self.logger.info(f"Case 2 features: {len(self.feature_names['case2'])} features")
            return df, None
    
    def engineer_case3_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering for Case 3: Shipping Delivery Days.
        CRITICAL: Only uses approved columns (leakage prevention).

        Per PDF specification:
        - Temporal features from order date (Order_Month, Order_DayOfWeek, Order_WeekOfYear)
        - One-hot encoding for: Shipping Mode, Type, Market, Order Region
        - Target encoding already applied in preprocessing for: Order Country, Order State

        Args:
            df: Preprocessed dataframe (already filtered to approved columns)

        Returns:
            Feature-engineered dataframe.
        """
        target = TARGET_COLUMNS['case3']
        date_col = DATE_COLUMNS['case3']

        df = df.copy()

        # 4. Temporal Feature Engineering (per PDF section 4)
        # Derived from order date (DateOrders)
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            # Create: Order_Month ∈ {1,…,12}
            df['Order_Month'] = df[date_col].dt.month
            # Create: Order_DayOfWeek ∈ {0,…,6}
            df['Order_DayOfWeek'] = df[date_col].dt.dayofweek
            # Create: Order_WeekOfYear ∈ {1,…,52}
            week = df[date_col].dt.isocalendar().week
            df['Order_WeekOfYear'] = week.fillna(0).astype(int)
            # Drop the date column after extraction
            df = df.drop(columns=[date_col])

        df = self.create_cyclical_features(df, 'Order_Month', 12, 'order_month')
        df = self.create_cyclical_features(df, 'Order_DayOfWeek', 7, 'order_dayofweek')
        df = self.create_cyclical_features(df, 'Order_WeekOfYear', 52, 'order_weekofyear')

        # Shipping mode ordinal and expedited flags
        if 'Shipping Mode' in df.columns:
            mode_rank = {
                'Same Day': 0,
                'First Class': 1,
                'Second Class': 2,
                'Standard Class': 3
            }
            df['shipping_mode_rank'] = df['Shipping Mode'].map(mode_rank).fillna(2).astype(int)
            df['is_expedited'] = df['Shipping Mode'].isin(['Same Day', 'First Class']).astype(int)

        # 3.1 Store categorical columns (encoding deferred to encode_for_model)
        # For: Shipping Mode, Type, Market, Order Region
        existing_onehot = [c for c in CASE3_ONEHOT_COLS if c in df.columns]
        self.cat_cols['case3'] = existing_onehot
        self.logger.info(f"Categorical columns stored for deferred encoding: {existing_onehot}")

        # Note: Target encoding for Order Country, Order State is done in preprocessing

        # Derived ratio features
        if 'Order Item Quantity' in df.columns and 'Order Item Total' in df.columns:
            denom = df['Order Item Quantity'].replace(0, np.nan)
            df['order_value_per_unit'] = (df['Order Item Total'] / denom).fillna(0)

        if 'Order Item Quantity' in df.columns and 'Sales' in df.columns:
            denom = df['Order Item Quantity'].replace(0, np.nan)
            df['sales_per_item'] = (df['Sales'] / denom).fillna(0)

        if 'Order Item Quantity' in df.columns and 'Order Item Discount' in df.columns:
            denom = df['Order Item Quantity'].replace(0, np.nan)
            df['discount_per_item'] = (df['Order Item Discount'] / denom).fillna(0)

        if 'Order Item Discount' in df.columns and 'Order Item Total' in df.columns:
            denom = df['Order Item Total'].replace(0, np.nan)
            df['discount_to_total'] = (df['Order Item Discount'] / denom).fillna(0)

        if 'Sales' in df.columns and 'Order Item Total' in df.columns:
            denom = df['Order Item Total'].replace(0, np.nan)
            df['sales_to_total'] = (df['Sales'] / denom).fillna(0)

        # Log transforms for skewed numeric features
        log_cols = [
            'Order Item Total',
            'Sales',
            'Product Price',
            'Order Item Discount'
        ]
        for col in log_cols:
            if col in df.columns:
                key = col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
                df[f'log_{key}'] = np.log1p(pd.to_numeric(df[col], errors='coerce').clip(lower=0))

        # Store feature names (excluding target)
        self.feature_names['case3'] = [c for c in df.columns if c != target]

        self.logger.info(f"Case 3 features: {len(self.feature_names['case3'])} features")
        return df
    
    def engineer_case4_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feature engineering for Case 4: Retail Store Inventory.
        
        Args:
            df: Preprocessed dataframe
            
        Returns:
            Feature-engineered dataframe.
        """
        target = TARGET_COLUMNS['case4']
        date_col = DATE_COLUMNS['case4']
        
        df = df.copy()
        if 'Demand Forecast' in df.columns:
            df = df.drop(columns=['Demand Forecast'])
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(['Store ID', 'Product ID', date_col])
        
        # Create lag features within each Store-Product panel
        lags = LAG_FEATURES['case4']
        for lag in lags:
            df[f'lag_{lag}'] = df.groupby(['Store ID', 'Product ID'])[target].shift(lag)
        
        # Rolling statistics
        windows = ROLLING_WINDOWS['case4']
        for window in windows:
            df[f'rolling_mean_{window}'] = df.groupby(['Store ID', 'Product ID'])[target].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).mean()
            )
            df[f'rolling_std_{window}'] = df.groupby(['Store ID', 'Product ID'])[target].transform(
                lambda x: x.shift(1).rolling(window, min_periods=1).std()
            )
        
        # Price-related features
        if 'Competitor Pricing' in df.columns and 'Price' in df.columns:
            df['price_ratio'] = df['Price'] / df['Competitor Pricing'].replace(0, 1)
            df['price_gap'] = df['Price'] - df['Competitor Pricing']
        
        if 'Price' in df.columns and 'Discount' in df.columns:
            df['price_after_discount'] = df['Price'] * (1 - df['Discount'] / 100)
            df['discount_amount'] = df['Price'] * (df['Discount'] / 100)

        if 'Inventory Level' in df.columns and 'Units Ordered' in df.columns:
            denom = df['Inventory Level'].replace(0, np.nan)
            df['inventory_to_order_ratio'] = (df['Units Ordered'] / denom).fillna(0)
            df['inventory_buffer'] = df['Inventory Level'] - df['Units Ordered']

        for col in ['Price', 'Competitor Pricing', 'Inventory Level', 'Units Ordered']:
            if col in df.columns:
                key = col.lower().replace(' ', '_')
                df[f'log_{key}'] = np.log1p(pd.to_numeric(df[col], errors='coerce').clip(lower=0))
        
        # Calendar features
        df = self.create_calendar_features(df, date_col)

        if date_col in df.columns:
            df['time_index'] = (df[date_col] - df[date_col].min()).dt.days
        df = self.create_cyclical_features(df, 'month', 12, 'month')
        df = self.create_cyclical_features(df, 'week_of_year', 52, 'week_of_year')
        df = self.create_cyclical_features(df, 'day_of_week', 7, 'day_of_week')
        df = self.create_cyclical_features(df, 'quarter', 4, 'quarter')
        
        # Store categorical columns (encoding deferred to encode_for_model)
        categorical_cols = [
            'Category',
            'Region',
            'Weather Condition',
            'Seasonality'
        ]
        self.cat_cols['case4'] = [c for c in categorical_cols if c in df.columns]
        
        # Drop rows with NaN from lag creation
        lag_cols = [f'lag_{lag}' for lag in lags]
        df = df.dropna(subset=lag_cols)
        
        # Store feature names
        self.feature_names['case4'] = [c for c in df.columns 
                                        if c not in [target, date_col, 'Store ID', 'Product ID']]
        
        self.logger.info(f"Case 4 features: {len(self.feature_names['case4'])} features")
        return df
    
    def get_feature_names(self, case: str) -> List[str]:
        """Get feature names for a specific case."""
        if case not in self.feature_names:
            raise ValueError(f"Features for '{case}' not yet created")
        return self.feature_names[case]
    
    def get_cat_cols(self, case: str) -> List[str]:
        """Get categorical column names for a specific case."""
        return self.cat_cols.get(case, [])
    
    def encode_for_model(self, df: pd.DataFrame, case: str,
                         model_type: str = 'default') -> Tuple[pd.DataFrame, List[str]]:
        """
        Apply model-specific encoding.
        
        For CatBoost: return raw categoricals + cat_features list.
        For others: apply one-hot encoding.
        
        Args:
            df: DataFrame with raw categorical columns
            case: Case identifier
            model_type: 'catboost' or 'default'
            
        Returns:
            Tuple of (encoded DataFrame, list of cat feature names/indices)
        """
        cat_cols = self.get_cat_cols(case)
        existing = [c for c in cat_cols if c in df.columns]
        
        if model_type == 'catboost':
            # Convert to string type for CatBoost native handling
            df_out = df.copy()
            for c in existing:
                df_out[c] = df_out[c].astype(str)
            return df_out, existing
        else:
            # One-hot encode for all other models
            if existing:
                df_out = pd.get_dummies(df, columns=existing, drop_first=True)
            else:
                df_out = df.copy()
            return df_out, []
    
    def split_case2_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split Case 2 data into train/validation/test sets based on week.
        
        Args:
            df: Full training dataframe (weeks 1-145)
            
        Returns:
            Tuple of (train, validation, test) dataframes.
        """
        train = df[(df['week'] >= CASE2_TRAIN_WEEKS[0]) & (df['week'] <= CASE2_TRAIN_WEEKS[1])]
        val = df[(df['week'] >= CASE2_VAL_WEEKS[0]) & (df['week'] <= CASE2_VAL_WEEKS[1])]
        test = df[(df['week'] >= CASE2_TEST_WEEKS[0]) & (df['week'] <= CASE2_TEST_WEEKS[1])]
        
        self.logger.info(f"Case 2 split: Train {len(train)}, Val {len(val)}, Test {len(test)}")
        return train, val, test
