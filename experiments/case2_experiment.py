"""Case 2: Food Demand Experiment (Competition-Style)."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent.parent))

from experiments.base_experiment import BaseExperiment
from config.config import Config
from config.constants import (
    TARGET_COLUMNS, CASE2_TRAIN_WEEKS, CASE2_VAL_WEEKS, 
    CASE2_TEST_WEEKS, RANDOM_SEED
)
from models.classical import NaiveForecaster, SARIMAXModel, ETSModel, ARIMAXModel, LinearRegressionModel
from models.ml import XGBoostModel, CatBoostModel, FFNNModel, RandomForestModel, ShallowANNModel


class Case2Experiment(BaseExperiment):
    """
    Experiment for Case 2: Food Demand (Competition-Style).
    
    Models: Seasonal Naive, SARIMA, XGBoost, FFNN
    Special: Uses 3-way split (train/val/test) and generates predictions for test file.
    """
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.target_col = TARGET_COLUMNS['case2']
        self.train_file = None
        self.test_file = None
        self.val_data = None
        self.X_val = None
        self.y_val = None
    
    @property
    def case_name(self) -> str:
        return 'case2'
    
    def load_data(self) -> pd.DataFrame:
        """Load Case 2 datasets (train and test files)."""
        self.train_file, self.test_file = self.data_loader.load_case2(include_test=True)
        self.raw_data = self.train_file.copy()
        self.logger.info(f"Train file: {len(self.train_file)} rows")
        if self.test_file is not None:
            self.logger.info(f"Test file: {len(self.test_file)} rows (for prediction)")
        return self.raw_data
    
    def preprocess(self) -> pd.DataFrame:
        """Preprocess Case 2 data."""
        self.processed_data = self.preprocessor.preprocess_case2(self.raw_data)
        return self.processed_data
    
    def engineer_features(self) -> pd.DataFrame:
        """Create features for Case 2."""
        # Engineer features (combines train and test for consistent encoding)
        self.processed_data, self.test_processed = self.feature_engineer.engineer_case2_features(
            self.train_file, 
            self.test_file
        )
        self.logger.info(f"Feature engineered data: {len(self.processed_data)} rows")
        return self.processed_data
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Week-based train/validation/test split for Case 2.
        
        Train: weeks 1-100
        Validation: weeks 101-125
        Test: weeks 126-145
        """
        df = self.processed_data.copy()
        
        # Split based on week
        self.train_data = df[(df['week'] >= CASE2_TRAIN_WEEKS[0]) & 
                            (df['week'] <= CASE2_TRAIN_WEEKS[1])].copy()
        self.val_data = df[(df['week'] >= CASE2_VAL_WEEKS[0]) & 
                          (df['week'] <= CASE2_VAL_WEEKS[1])].copy()
        self.test_data = df[(df['week'] >= CASE2_TEST_WEEKS[0]) & 
                           (df['week'] <= CASE2_TEST_WEEKS[1])].copy()
        
        # Get feature columns (exclude identifiers and targets)
        # Exclude identifier columns (v3 has no center_id/meal_id)
        possible_id_cols = ['center_id', 'meal_id', 'city_code']
        id_cols_present = [c for c in possible_id_cols if c in self.processed_data.columns]
        exclude_cols = [self.target_col, 'log_num_orders', 'week'] + id_cols_present
        feature_cols = [c for c in self.train_data.columns if c not in exclude_cols]
        
        # Get categorical columns from feature engineer
        cat_cols = self.feature_engineer.get_cat_cols('case2')
        cat_cols_present = [c for c in cat_cols if c in feature_cols]
        
        # Numeric feature columns (for non-CatBoost models)
        numeric_cols = [c for c in feature_cols 
                       if self.train_data[c].dtype in ['float64', 'int64', 'float32', 'int32', 'uint8']]
        
        self.feature_cols = numeric_cols
        self._cat_cols = cat_cols_present
        
        # Drop NaN rows (use numeric cols + target for NaN check)
        check_cols = numeric_cols + [self.target_col]
        self.train_data = self.train_data.dropna(subset=check_cols)
        self.val_data = self.val_data.dropna(subset=check_cols)
        self.test_data = self.test_data.dropna(subset=check_cols)
        
        # Encoded feature sets for LR/RF/XGB/NN (one-hot encoded)
        train_encoded, _ = self.feature_engineer.encode_for_model(
            self.train_data[numeric_cols + cat_cols_present], 'case2', 'default')
        val_encoded, _ = self.feature_engineer.encode_for_model(
            self.val_data[numeric_cols + cat_cols_present], 'case2', 'default')
        test_encoded, _ = self.feature_engineer.encode_for_model(
            self.test_data[numeric_cols + cat_cols_present], 'case2', 'default')
        
        # Align columns after one-hot encoding
        common_cols = sorted(set(train_encoded.columns) & set(val_encoded.columns) & set(test_encoded.columns))
        self.X_train = train_encoded[common_cols]
        self.X_val = val_encoded[common_cols]
        self.X_test = test_encoded[common_cols]
        
        # Raw feature sets for CatBoost (with categorical columns)
        cat_train, self._cat_feature_names = self.feature_engineer.encode_for_model(
            self.train_data[numeric_cols + cat_cols_present], 'case2', 'catboost')
        cat_val, _ = self.feature_engineer.encode_for_model(
            self.val_data[numeric_cols + cat_cols_present], 'case2', 'catboost')
        cat_test, _ = self.feature_engineer.encode_for_model(
            self.test_data[numeric_cols + cat_cols_present], 'case2', 'catboost')
        self.X_train_cat = cat_train
        self.X_val_cat = cat_val
        self.X_test_cat = cat_test
        
        # Use log-transformed target for training
        self.y_train = self.train_data['log_num_orders'].values
        self.y_val = self.val_data['log_num_orders'].values
        self.y_test = self.test_data['log_num_orders'].values
        
        # Keep original target for evaluation
        self.y_test_original = self.test_data[self.target_col].values
        
        self.logger.info(f"Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")
        self.logger.info(f"Encoded features: {len(common_cols)}, CatBoost raw features: {len(cat_train.columns)} (cats: {len(self._cat_feature_names)})")
        return self.train_data, self.test_data
    
    def train_classical_models(self) -> Dict:
        """Train classical statistical models for Case 2."""
        
        # Aggregate data for time series models
        train_agg = self.train_data.groupby('week')[self.target_col].sum().values
        
        # Build exogenous features for ARIMAX/SARIMAX
        # Aggregate numeric features at weekly level
        exog_candidates = ['checkout_price', 'base_price', 'discount_percent',
                          'emailer_for_promotion', 'homepage_featured', 'promo_count']
        exog_cols = [c for c in exog_candidates if c in self.train_data.columns]
        if exog_cols:
            exog_train_df = self.train_data.groupby('week')[exog_cols].mean()
            exog_train = exog_train_df.values
            test_agg_df = self.test_data.groupby('week')
            exog_test = test_agg_df[exog_cols].mean().values
        else:
            exog_train = None
            exog_test = None
        self._exog_test = exog_test
        
        # 1. Naive (non-seasonal)
        try:
            naive = NaiveForecaster(seasonal=False)
            naive.fit(train_agg)
            self.models['Naive'] = naive
            self.logger.info("Naive model trained")
        except Exception as e:
            self.logger.error(f"Naive failed: {e}")

        # 2. Seasonal Naive
        try:
            seasonal_naive = NaiveForecaster(seasonal=True, seasonal_period=4)
            seasonal_naive.fit(train_agg)
            self.models['SeasonalNaive'] = seasonal_naive
            self.logger.info("Seasonal Naive model trained")
        except Exception as e:
            self.logger.error(f"Seasonal Naive failed: {e}")

        # 3. ETS
        try:
            ets = ETSModel.auto_fit(train_agg, seasonal_periods=4)
            self.models['ETS'] = ets
            self.logger.info("ETS model trained")
        except Exception as e:
            self.logger.error(f"ETS failed: {e}")

        # 4. ARIMAX
        try:
            arimax = ARIMAXModel(auto_order=True, max_p=3, max_q=3)
            arimax.fit(train_agg, exog=exog_train)
            self.models['ARIMAX'] = arimax
            self.logger.info("ARIMAX model trained")
        except Exception as e:
            self.logger.error(f"ARIMAX failed: {e}")

        # 5. SARIMAX
        try:
            sarimax = SARIMAXModel(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 4),
                auto_order=False
            )
            sarimax.fit(train_agg, exog=exog_train)
            self.models['SARIMAX'] = sarimax
            self.logger.info("SARIMAX model trained")
        except Exception as e:
            self.logger.error(f"SARIMAX failed: {e}")
        
        return self.models
    
    def train_ml_models(self) -> Dict:
        """Train ML models for Case 2."""

        # 1. Linear Regression (log-target)
        try:
            lr = LinearRegressionModel()
            lr.fit(self.X_train, self.y_train)
            self.models['LinearRegression'] = lr
            self.logger.info("Linear Regression trained")
        except Exception as e:
            self.logger.error(f"Linear Regression failed: {e}")

        # 2. Random Forest
        try:
            rf = RandomForestModel()
            rf.fit(self.X_train, self.y_train)
            self.models['RandomForest'] = rf
            self.logger.info("Random Forest trained")
        except Exception as e:
            self.logger.error(f"Random Forest failed: {e}")

        # 3. XGBoost with early stopping using validation set
        try:
            xgb = XGBoostModel()
            xgb.fit(
                self.X_train, self.y_train,
                X_val=self.X_val, y_val=self.y_val,
                early_stopping_rounds=50
            )
            self.models['XGBoost'] = xgb
            self.logger.info("XGBoost trained")
        except Exception as e:
            self.logger.error(f"XGBoost failed: {e}")

        # 4. CatBoost with early stopping — uses RAW categorical features
        try:
            cat = CatBoostModel()
            cat.fit(
                self.X_train_cat, self.y_train,
                X_val=self.X_val_cat, y_val=self.y_val,
                early_stopping_rounds=50,
                cat_features=self._cat_feature_names
            )
            self.models['CatBoost'] = cat
            self.logger.info("CatBoost trained (with native categorical features)")
        except Exception as e:
            self.logger.error(f"CatBoost failed: {e}")

        # 5. FFNN and Shallow ANN
        if self.config.run_neural_networks:
            try:
                ffnn = FFNNModel(
                    hidden_layers=[128, 64, 32],
                    epochs=200,
                    patience=20
                )
                ffnn.fit(
                    self.X_train, self.y_train,
                    X_val=self.X_val, y_val=self.y_val,
                    verbose=0
                )
                self.models['FFNN'] = ffnn
                self.logger.info("FFNN trained")
            except Exception as e:
                self.logger.error(f"FFNN failed: {e}")

            try:
                shallow = ShallowANNModel()
                shallow.fit(
                    self.X_train, self.y_train,
                    X_val=self.X_val, y_val=self.y_val,
                    verbose=0
                )
                self.models['ShallowANN'] = shallow
                self.logger.info("Shallow ANN trained")
            except Exception as e:
                self.logger.error(f"Shallow ANN failed: {e}")
        
        return self.models
    
    def evaluate_models(self) -> pd.DataFrame:
        """Evaluate all models for Case 2 using RMSLE."""
        results_list = []
        
        for model_name, model in self.models.items():
            try:
                if model_name in ['Naive', 'SeasonalNaive', 'ETS']:
                    # Time series models - no exog
                    n_test_weeks = len(self.test_data['week'].unique())
                    preds_agg = model.predict(n_test_weeks)
                    
                    test_agg = self.test_data.groupby('week')[self.target_col].sum().values
                    
                    metrics = self.metrics.evaluate_model(
                        model_name=model_name,
                        y_true=test_agg,
                        y_pred=preds_agg,
                        case=self.case_name,
                        dataset_split='test',
                        include_rmsle=True
                    )
                    
                    self.predictions[model_name] = preds_agg
                
                elif model_name in ['ARIMAX', 'SARIMAX']:
                    # Time series models with exogenous variables
                    n_test_weeks = len(self.test_data['week'].unique())
                    exog_future = getattr(self, '_exog_test', None)
                    preds_agg = model.predict(n_test_weeks, exog_future=exog_future)
                    
                    test_agg = self.test_data.groupby('week')[self.target_col].sum().values
                    
                    metrics = self.metrics.evaluate_model(
                        model_name=model_name,
                        y_true=test_agg,
                        y_pred=preds_agg,
                        case=self.case_name,
                        dataset_split='test',
                        include_rmsle=True
                    )
                    
                    self.predictions[model_name] = preds_agg
                
                elif model_name == 'CatBoost':
                    # CatBoost uses raw categorical features
                    preds_log = model.predict(self.X_test_cat)
                    preds = np.expm1(preds_log)
                    preds = np.maximum(preds, 0)
                    
                    self.predictions[model_name] = preds
                    
                    metrics = self.metrics.evaluate_model(
                        model_name=model_name,
                        y_true=self.y_test_original,
                        y_pred=preds,
                        case=self.case_name,
                        dataset_split='test',
                        include_rmsle=True
                    )
                    
                else:
                    # ML models - predict log values, then convert back
                    preds_log = model.predict(self.X_test)
                    preds = np.expm1(preds_log)  # Reverse log1p
                    preds = np.maximum(preds, 0)  # Ensure non-negative
                    
                    self.predictions[model_name] = preds
                    
                    metrics = self.metrics.evaluate_model(
                        model_name=model_name,
                        y_true=self.y_test_original,
                        y_pred=preds,
                        case=self.case_name,
                        dataset_split='test',
                        include_rmsle=True
                    )
                
                results_list.append({
                    'model': model_name,
                    **metrics
                })
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
        
        self.results = pd.DataFrame(results_list)
        return self.results
    
    def generate_predictions_for_test_file(self) -> pd.DataFrame:
        """
        Generate predictions for the actual test file (weeks 146-155).

        Uses XGBoost or FFNN for row-level predictions (time series models
        predict at week level which doesn't match test file granularity).

        Returns:
            Test file with num_orders column added
        """
        if self.test_file is None or self.test_processed is None:
            self.logger.warning("No test file loaded for prediction")
            return None

        # For test file predictions, we need ML models that predict at row level
        # Time series models (SeasonalNaive, SARIMA) predict aggregated weekly totals
        ml_models = ['XGBoost', 'FFNN', 'ShallowANN', 'RandomForest', 'LinearRegression']
        best_model = None
        best_model_name = None

        # Find best ML model based on R2 (row-level performance)
        for model_name in ml_models:
            if model_name in self.models:
                best_model_name = model_name
                best_model = self.models[model_name]
                break

        if best_model is None:
            self.logger.error("No ML models available for row-level predictions")
            return None

        self.logger.info(f"Using {best_model_name} for final predictions")

        # Prepare test features
        test_features = self.test_processed[self.feature_cols].copy()

        # Fill any NaN values with 0 (for lag features)
        test_features = test_features.fillna(0)

        # Generate predictions (ML models predict log-transformed target)
        preds_log = best_model.predict(test_features)
        preds = np.expm1(preds_log)
        preds = np.maximum(preds, 0).astype(int)

        # Add predictions to test file
        result = self.test_file.copy()
        result['num_orders'] = preds[:len(result)]

        # Save predictions
        output_path = self.config.results_dir / 'case2_predictions.csv'
        result.to_csv(output_path, index=False)
        self.logger.info(f"Predictions saved to {output_path}")

        return result
    
    def generate_figures(self):
        """Generate figures for Case 2."""
        super().generate_figures()
        
        # Figure 6.2: Time series plot
        try:
            agg_data = self.processed_data.groupby('week')[self.target_col].sum().reset_index()
            self.visualizer.plot_time_series(
                data=agg_data,
                date_col='week',
                value_col=self.target_col,
                title='Food Demand Over Time (Aggregated)',
                filename='figure_6_2_case2_timeseries',
                chapter='chapter6'
            )
        except Exception as e:
            self.logger.error(f"Figure 6.2 failed: {e}")
        
        # Figure 6.5: Distribution plot
        try:
            self.visualizer.plot_distribution(
                data=self.processed_data[self.target_col].dropna(),
                title='Distribution of Food Orders',
                filename='figure_6_5_case2_distribution',
                xlabel='Number of Orders',
                chapter='chapter6'
            )
        except Exception as e:
            self.logger.error(f"Figure 6.5 failed: {e}")
        
        # Figure 7.3: Seasonal Decomposition
        try:
            agg_series = self.processed_data.groupby('week')[self.target_col].sum()
            self.visualizer.plot_seasonal_decomposition(
                data=agg_series,
                title='Seasonal Decomposition of Food Demand',
                filename='figure_7_3_case2_decomposition',
                period=4,
                chapter='chapter7'
            )
        except Exception as e:
            self.logger.error(f"Figure 7.3 failed: {e}")

        # Figure 7.4: Peak season forecast vs actual (SARIMA vs ML models)
        if 'SARIMA' in self.models and ('XGBoost' in self.models or 'FFNN' in self.models):
            try:
                # Identify peak periods (top 25% by demand)
                test_agg = self.test_data.groupby('week').agg({
                    self.target_col: 'sum'
                }).reset_index()
                threshold = test_agg[self.target_col].quantile(0.75)
                peak_mask = test_agg[self.target_col] >= threshold

                # Get aggregated predictions for comparison
                predictions_for_plot = {}
                if 'XGBoost' in self.predictions:
                    # Aggregate XGBoost predictions by week
                    test_with_preds = self.test_data.copy()
                    test_with_preds['xgb_pred'] = self.predictions['XGBoost']
                    xgb_agg = test_with_preds.groupby('week')['xgb_pred'].sum().values
                    predictions_for_plot['XGBoost'] = xgb_agg

                if 'FFNN' in self.predictions:
                    test_with_preds = self.test_data.copy()
                    test_with_preds['ffnn_pred'] = self.predictions['FFNN']
                    ffnn_agg = test_with_preds.groupby('week')['ffnn_pred'].sum().values
                    predictions_for_plot['FFNN'] = ffnn_agg

                if predictions_for_plot:
                    self.visualizer.plot_peak_season_comparison(
                        data=test_agg,
                        time_col='week',
                        actual_col=self.target_col,
                        predictions=predictions_for_plot,
                        peak_mask=peak_mask.values,
                        title='Forecast vs Actual During Peak Seasons (Food Demand)',
                        filename='figure_7_4_case2_peak_season',
                        chapter='chapter7'
                    )
            except Exception as e:
                self.logger.error(f"Figure 7.4 failed: {e}")

        # Figure 7.5: Distribution of Forecast Errors (KDE per PDF specification)
        try:
            # Only include ML models that predict at row level (not aggregated)
            errors = {}
            for model_name in ['XGBoost', 'FFNN', 'ShallowANN', 'RandomForest', 'LinearRegression']:
                if model_name in self.predictions:
                    preds = self.predictions[model_name]
                    if len(preds) == len(self.y_test_original):
                        errors[model_name] = self.y_test_original - preds

            if errors:
                self.visualizer.plot_error_distribution_kde(
                    errors=errors,
                    title='Distribution of Forecast Errors (Case 2 - Food Demand)',
                    filename='figure_7_5_case2_error_distribution',
                    chapter='chapter7'
                )
        except Exception as e:
            self.logger.error(f"Figure 7.5 failed: {e}")

        # Figure 7.11: Feature Importance (XGBoost)
        if 'XGBoost' in self.models:
            try:
                importance_df = self.models['XGBoost'].get_feature_importance(
                    top_n=15,
                    feature_names=getattr(self, 'feature_cols', None)
                )
                self.visualizer.plot_feature_importance(
                    importance_df=importance_df,
                    title='Feature Importance for Food Demand (XGBoost)',
                    filename='figure_7_11_case2_importance',
                    chapter='chapter7'
                )
            except Exception as e:
                self.logger.error(f"Feature importance figure failed: {e}")
    
    def run(self, save_results: bool = True) -> pd.DataFrame:
        """Run the complete Case 2 experiment."""
        results = super().run(save_results)
        
        # Generate final predictions for test file
        if self.test_file is not None:
            self.generate_predictions_for_test_file()
        
        return results


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    config = Config()
    experiment = Case2Experiment(config)
    results = experiment.run()
    
    print("\nCase 2 Results:")
    print(results.to_string())
