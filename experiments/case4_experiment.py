"""Case 4: Retail Store Inventory Experiment."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).parent.parent))

from experiments.base_experiment import BaseExperiment
from config.config import Config
from config.constants import TARGET_COLUMNS, DATE_COLUMNS, TRAIN_RATIO, RANDOM_SEED
from models.classical import ETSModel, ARIMAXModel, SARIMAXModel, NaiveForecaster, LinearRegressionModel
from models.ml import XGBoostModel, CatBoostModel, RandomForestModel, FFNNModel, ShallowANNModel


class Case4Experiment(BaseExperiment):
    """
    Experiment for Case 4: Retail Store Inventory.
    
    Models: ETS, Linear Regression, XGBoost
    Panel data: Store ID x Product ID x Date
    """
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.target_col = TARGET_COLUMNS['case4']
        self.date_col = DATE_COLUMNS['case4']
    
    @property
    def case_name(self) -> str:
        return 'case4'
    
    def load_data(self) -> pd.DataFrame:
        """Load Case 4 dataset."""
        self.raw_data = self.data_loader.load_case4()
        self.logger.info(f"Loaded {len(self.raw_data)} rows")
        return self.raw_data
    
    def preprocess(self) -> pd.DataFrame:
        """Preprocess Case 4 data."""
        self.processed_data = self.preprocessor.preprocess_case4(self.raw_data)
        return self.processed_data
    
    def engineer_features(self) -> pd.DataFrame:
        """Create features for Case 4."""
        self.processed_data = self.feature_engineer.engineer_case4_features(self.processed_data)
        self.logger.info(f"Feature engineered data: {len(self.processed_data)} rows")
        return self.processed_data
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Time-based train/test split for Case 4."""
        df = self.processed_data.copy()
        df = df.sort_values(self.date_col)
        
        # Time-based split
        split_idx = int(len(df) * TRAIN_RATIO)
        
        self.train_data = df.iloc[:split_idx].copy()
        self.test_data = df.iloc[split_idx:].copy()
        
        # Get feature columns
        feature_cols = self.feature_engineer.get_feature_names('case4')
        
        # Get categorical columns
        cat_cols = self.feature_engineer.get_cat_cols('case4')
        cat_cols_present = [c for c in cat_cols 
                           if c in self.train_data.columns and c in feature_cols]
        
        # Numeric feature columns
        numeric_cols = [c for c in feature_cols 
                       if c in self.train_data.columns and c not in cat_cols_present and
                       self.train_data[c].dtype in ['float64', 'int64', 'float32', 'int32', 'uint8']]
        
        self.feature_cols = numeric_cols
        self._cat_cols = cat_cols_present
        
        # Drop NaN rows
        self.train_data = self.train_data.dropna(subset=numeric_cols + [self.target_col])
        self.test_data = self.test_data.dropna(subset=numeric_cols + [self.target_col])
        
        # Encoded features for LR/RF/XGB/NN (one-hot)
        train_encoded, _ = self.feature_engineer.encode_for_model(
            self.train_data[numeric_cols + cat_cols_present], 'case4', 'default')
        test_encoded, _ = self.feature_engineer.encode_for_model(
            self.test_data[numeric_cols + cat_cols_present], 'case4', 'default')
        
        common_cols = sorted(set(train_encoded.columns) & set(test_encoded.columns))
        self.X_train = train_encoded[common_cols]
        self.X_test = test_encoded[common_cols]
        
        # Raw features for CatBoost
        cat_train, self._cat_feature_names = self.feature_engineer.encode_for_model(
            self.train_data[numeric_cols + cat_cols_present], 'case4', 'catboost')
        cat_test, _ = self.feature_engineer.encode_for_model(
            self.test_data[numeric_cols + cat_cols_present], 'case4', 'catboost')
        self.X_train_cat = cat_train
        self.X_test_cat = cat_test
        
        self.y_train = self.train_data[self.target_col].values
        self.y_test = self.test_data[self.target_col].values
        
        self.logger.info(f"Train: {len(self.train_data)}, Test: {len(self.test_data)}")
        self.logger.info(f"Encoded: {len(common_cols)}, CatBoost raw: {len(cat_train.columns)} (cats: {len(self._cat_feature_names)})")
        return self.train_data, self.test_data
    
    def train_classical_models(self) -> Dict:
        """Train classical statistical models for Case 4."""
        
        # Get aggregated time series for ETS
        train_agg = self.train_data.groupby(self.date_col)[self.target_col].sum()
        
        # Build exogenous features for ARIMAX/SARIMAX
        exog_candidates = ['Price', 'Discount', 'Competitor Pricing',
                          'month', 'quarter', 'month_sin', 'month_cos', 'time_index']
        exog_cols = [c for c in exog_candidates if c in self.train_data.columns]
        if exog_cols:
            exog_train = self.train_data.groupby(self.date_col)[exog_cols].mean().values
            exog_test = self.test_data.groupby(self.date_col)[exog_cols].mean().values
        else:
            exog_train = None
            exog_test = None
        self._exog_test = exog_test
        
        # 1. Naive (non-seasonal)
        try:
            naive = NaiveForecaster(seasonal=False)
            naive.fit(train_agg.values)
            self.models['Naive'] = naive
            self.logger.info("Naive model trained")
        except Exception as e:
            self.logger.error(f"Naive failed: {e}")

        # 2. Seasonal Naive
        try:
            seasonal_naive = NaiveForecaster(seasonal=True, seasonal_period=7)
            seasonal_naive.fit(train_agg.values)
            self.models['SeasonalNaive'] = seasonal_naive
            self.logger.info("Seasonal Naive model trained")
        except Exception as e:
            self.logger.error(f"Seasonal Naive failed: {e}")

        # 3. Exponential Smoothing (ETS)
        try:
            ets = ETSModel.auto_fit(train_agg.values, seasonal_periods=7)
            self.models['ETS'] = ets
            self.logger.info("ETS model trained")
        except Exception as e:
            self.logger.error(f"ETS failed: {e}")

        # 4. ARIMAX
        try:
            arimax = ARIMAXModel(auto_order=True, max_p=3, max_q=3)
            arimax.fit(train_agg.values, exog=exog_train)
            self.models['ARIMAX'] = arimax
            self.logger.info("ARIMAX model trained")
        except Exception as e:
            self.logger.error(f"ARIMAX failed: {e}")

        # 5. SARIMAX
        try:
            sarimax = SARIMAXModel(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                auto_order=False
            )
            sarimax.fit(train_agg.values, exog=exog_train)
            self.models['SARIMAX'] = sarimax
            self.logger.info("SARIMAX model trained")
        except Exception as e:
            self.logger.error(f"SARIMAX failed: {e}")

        # 6. Linear Regression with lag features
        try:
            lr = LinearRegressionModel()
            lr.fit(self.X_train, self.y_train)
            self.models['LinearRegression'] = lr
            self.logger.info("Linear Regression trained")
        except Exception as e:
            self.logger.error(f"Linear Regression failed: {e}")
        
        return self.models
    
    def train_ml_models(self) -> Dict:
        """Train ML models for Case 4."""
        
        # Create validation split for neural nets and XGBoost
        X_t, X_v, y_t, y_v = train_test_split(
            self.X_train, self.y_train,
            test_size=0.2, random_state=RANDOM_SEED
        )

        # 1. Random Forest
        try:
            rf = RandomForestModel()
            rf.fit(self.X_train, self.y_train)
            self.models['RandomForest'] = rf
            self.logger.info("Random Forest trained")
        except Exception as e:
            self.logger.error(f"Random Forest failed: {e}")

        # 2. XGBoost
        try:
            xgb = XGBoostModel()
            xgb.fit(X_t, y_t, X_val=X_v, y_val=y_v)
            self.models['XGBoost'] = xgb
            self.logger.info("XGBoost trained")
        except Exception as e:
            self.logger.error(f"XGBoost failed: {e}")

        # 3. CatBoost — uses RAW categorical features
        try:
            cat = CatBoostModel()
            # Use raw cat features for CatBoost
            X_t_cat = self.X_train_cat.copy()
            X_v_cat = None  # Will create val split if needed
            cat.fit(
                self.X_train_cat, self.y_train,
                cat_features=self._cat_feature_names
            )
            self.models['CatBoost'] = cat
            self.logger.info("CatBoost trained (with native categorical features)")
        except Exception as e:
            self.logger.error(f"CatBoost failed: {e}")

        # 4. FFNN and Shallow ANN
        if self.config.run_neural_networks:
            try:
                ffnn = FFNNModel(
                    hidden_layers=[128, 64, 32],
                    epochs=200,
                    patience=20
                )
                ffnn.fit(X_t, y_t, X_val=X_v, y_val=y_v, verbose=0)
                self.models['FFNN'] = ffnn
                self.logger.info("FFNN trained")
            except Exception as e:
                self.logger.error(f"FFNN failed: {e}")

            try:
                shallow = ShallowANNModel()
                shallow.fit(X_t, y_t, X_val=X_v, y_val=y_v, verbose=0)
                self.models['ShallowANN'] = shallow
                self.logger.info("Shallow ANN trained")
            except Exception as e:
                self.logger.error(f"Shallow ANN failed: {e}")
        
        return self.models
    
    def evaluate_models(self) -> pd.DataFrame:
        """Evaluate all models for Case 4."""
        results_list = []
        
        for model_name, model in self.models.items():
            try:
                if model_name in ['Naive', 'SeasonalNaive', 'ETS']:
                    # Time series model - aggregate predictions
                    test_agg = self.test_data.groupby(self.date_col)[self.target_col].sum().values
                    n_periods = len(test_agg)
                    preds = model.predict(n_periods)
                    
                    metrics = self.metrics.evaluate_model(
                        model_name=model_name,
                        y_true=test_agg,
                        y_pred=preds,
                        case=self.case_name,
                        dataset_split='test'
                    )
                    
                    self.predictions[model_name] = preds
                elif model_name in ['ARIMAX', 'SARIMAX']:
                    test_agg = self.test_data.groupby(self.date_col)[self.target_col].sum().values
                    n_periods = len(test_agg)
                    exog_future = getattr(self, '_exog_test', None)
                    preds = model.predict(n_periods, exog_future=exog_future)
                    
                    metrics = self.metrics.evaluate_model(
                        model_name=model_name,
                        y_true=test_agg,
                        y_pred=preds,
                        case=self.case_name,
                        dataset_split='test'
                    )
                    
                    self.predictions[model_name] = preds
                elif model_name == 'CatBoost':
                    preds = model.predict(self.X_test_cat)
                    self.predictions[model_name] = preds
                    
                    metrics = self.metrics.evaluate_model(
                        model_name=model_name,
                        y_true=self.y_test,
                        y_pred=preds,
                        case=self.case_name,
                        dataset_split='test'
                    )
                else:
                    # Supervised models (encoded features)
                    preds = model.predict(self.X_test)
                    self.predictions[model_name] = preds
                    
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
        """Generate figures for Case 4."""
        super().generate_figures()
        
        # Figure 6.9: Time series plot (aggregated) - consistent naming with other cases
        try:
            agg_data = self.processed_data.groupby(self.date_col)[self.target_col].sum().reset_index()
            self.visualizer.plot_time_series(
                data=agg_data,
                date_col=self.date_col,
                value_col=self.target_col,
                title='Retail Store Units Sold Over Time',
                filename='figure_6_9_case4_timeseries',
                chapter='chapter6'
            )
        except Exception as e:
            self.logger.error(f"Figure 6.9 failed: {e}")

        # Figure 6.10: Distribution of units sold (Case 4)
        try:
            self.visualizer.plot_distribution(
                data=self.processed_data[self.target_col].dropna(),
                title='Distribution of Units Sold',
                filename='figure_6_10_case4_distribution',
                xlabel='Units Sold',
                chapter='chapter6'
            )
        except Exception as e:
            self.logger.error(f"Figure 6.10 failed: {e}")
        
        # Figure 7.5: Actual vs Predicted
        if 'XGBoost' in self.predictions:
            try:
                self.visualizer.plot_actual_vs_predicted(
                    y_true=self.y_test,
                    y_pred=self.predictions['XGBoost'],
                    model_name='XGBoost',
                    title='Actual vs Predicted Units Sold (Case 4)',
                    filename='figure_7_5_case4_xgb_scatter',
                    chapter='chapter7'
                )
            except Exception as e:
                self.logger.error(f"Figure 7.5 failed: {e}")
        
        # Figure 7.6: Boxplot of Absolute Errors (per PDF specification - boxplot format)
        try:
            errors = {
                model: np.abs(self.y_test - preds)
                for model, preds in self.predictions.items()
                if model != 'ETS' and len(preds) == len(self.y_test)
            }
            if errors:
                self.visualizer.plot_error_boxplot(
                    errors=errors,
                    title='Distribution of Absolute Prediction Errors (Case 4 - Retail)',
                    filename='figure_7_6_case4_error_boxplot',
                    chapter='chapter7'
                )
        except Exception as e:
            self.logger.error(f"Figure 7.6 failed: {e}")

        # Figure 7.11: Feature Importance (XGBoost)
        if 'XGBoost' in self.models:
            try:
                importance_df = self.models['XGBoost'].get_feature_importance(
                    feature_names=self.feature_cols,
                    top_n=15
                )
                self.visualizer.plot_feature_importance(
                    importance_df=importance_df,
                    title='Feature Importance for Retail Demand (XGBoost)',
                    filename='figure_7_11_case4_importance',
                    chapter='chapter7'
                )
            except Exception as e:
                self.logger.error(f"Feature importance figure failed: {e}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    config = Config()
    experiment = Case4Experiment(config)
    results = experiment.run()
    
    print("\nCase 4 Results:")
    print(results.to_string())
