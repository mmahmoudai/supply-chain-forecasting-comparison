"""Case 3: Shipping Delivery Days Experiment."""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(str(Path(__file__).parent.parent))

from experiments.base_experiment import BaseExperiment
from config.config import Config
from config.constants import TARGET_COLUMNS, DATE_COLUMNS, TRAIN_RATIO, RANDOM_SEED
from models.classical import LinearRegressionModel, NaiveForecaster, ETSModel, ARIMAXModel, SARIMAXModel
from models.ml import RandomForestModel, XGBoostModel, CatBoostModel, FFNNModel, ShallowANNModel


class Case3Experiment(BaseExperiment):
    """
    Experiment for Case 3: Shipping Delivery Days.
    
    Models: Linear Regression, Random Forest, XGBoost
    CRITICAL: Implements leakage prevention by using only approved columns.
    """
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.target_col = TARGET_COLUMNS['case3']
        self.date_col = DATE_COLUMNS['case3']
        self.scaler = StandardScaler()
        self.ts_train = None
        self.ts_test = None
    
    @property
    def case_name(self) -> str:
        return 'case3'
    
    def load_data(self) -> pd.DataFrame:
        """Load Case 3 dataset."""
        self.raw_data = self.data_loader.load_case3()
        self.logger.info(f"Loaded {len(self.raw_data)} rows, {len(self.raw_data.columns)} columns")
        return self.raw_data
    
    def preprocess(self) -> pd.DataFrame:
        """
        Preprocess Case 3 data with LEAKAGE PREVENTION.
        
        This is critical - we only keep approved columns.
        """
        self.processed_data = self.preprocessor.preprocess_case3(self.raw_data)
        self.logger.info(f"After preprocessing: {len(self.processed_data.columns)} columns (leakage-free)")
        return self.processed_data
    
    def engineer_features(self) -> pd.DataFrame:
        """Create features for Case 3."""
        self.processed_data = self.feature_engineer.engineer_case3_features(self.processed_data)
        self.logger.info(f"Feature engineered data: {len(self.processed_data)} rows, {len(self.processed_data.columns)} columns")
        return self.processed_data
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Random train/test split for Case 3 (cross-sectional data)."""
        df = self.processed_data.copy()
        
        # Get feature columns
        feature_cols = self.feature_engineer.get_feature_names('case3')
        
        # Get categorical columns from feature engineer
        cat_cols = self.feature_engineer.get_cat_cols('case3')
        cat_cols_present = [c for c in cat_cols if c in df.columns and c in feature_cols]
        
        # Numeric feature columns
        numeric_cols = [c for c in feature_cols 
                       if c in df.columns and c not in cat_cols_present and
                       df[c].dtype in ['float64', 'int64', 'float32', 'int32', 'uint8']]
        
        self.feature_cols = numeric_cols
        self._cat_cols = cat_cols_present
        
        # Remove rows with NaN
        df = df.dropna(subset=numeric_cols + [self.target_col])
        
        # --- CatBoost feature set (unscaled, with raw categoricals) ---
        catboost_cols = numeric_cols + cat_cols_present
        X_cat_full = df[catboost_cols].copy()
        for c in cat_cols_present:
            X_cat_full[c] = X_cat_full[c].astype(str)
        
        y_full = df[self.target_col].values
        
        # Random split
        X_num = df[numeric_cols]
        X_train_num, X_test_num, y_train, y_test = train_test_split(
            X_num, y_full, test_size=(1 - TRAIN_RATIO), random_state=RANDOM_SEED
        )
        
        # Scale numeric features for non-CatBoost models
        self.X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train_num),
            columns=numeric_cols,
            index=X_train_num.index
        )
        self.X_test = pd.DataFrame(
            self.scaler.transform(X_test_num),
            columns=numeric_cols,
            index=X_test_num.index
        )
        
        # CatBoost feature sets (unscaled + raw categoricals)
        self.X_train_cat = X_cat_full.loc[X_train_num.index].copy()
        self.X_test_cat = X_cat_full.loc[X_test_num.index].copy()
        self._cat_feature_names = cat_cols_present
        
        self.y_train = y_train
        self.y_test = y_test
        
        # Store full dataframes for reference
        self.train_data = df.loc[X_train_num.index]
        self.test_data = df.loc[X_test_num.index]
        
        self.logger.info(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        self.logger.info(f"CatBoost features: {len(catboost_cols)} (cats: {len(cat_cols_present)})")
        self._prepare_time_series()
        return self.train_data, self.test_data

    def _prepare_time_series(self) -> None:
        """Prepare aggregated daily series for time-series models."""
        if self.raw_data is None:
            self.logger.warning("Raw data not available for time-series preparation")
            return
        if self.date_col not in self.raw_data.columns:
            self.logger.warning("Date column missing for time-series preparation")
            return

        ts_data = self.raw_data[[self.date_col, self.target_col]].copy()
        ts_data[self.date_col] = pd.to_datetime(ts_data[self.date_col], errors='coerce')
        ts_data = ts_data.dropna(subset=[self.date_col, self.target_col])
        if ts_data.empty:
            self.logger.warning("No valid rows for time-series preparation")
            return

        ts_data['date_only'] = ts_data[self.date_col].dt.date
        ts_series = ts_data.groupby('date_only')[self.target_col].mean().sort_index()
        if ts_series.empty:
            self.logger.warning("No aggregated series available for time-series models")
            return

        values = ts_series.values
        split_idx = int(len(values) * TRAIN_RATIO)
        self.ts_train = values[:split_idx]
        self.ts_test = values[split_idx:]
        self.logger.info(f"Time-series split: Train {len(self.ts_train)}, Test {len(self.ts_test)}")
    
    def train_classical_models(self) -> Dict:
        """Train classical models for Case 3."""

        # Time-series models on aggregated daily series
        if self.ts_train is not None and len(self.ts_train) > 1:
            # Build minimal exog for ARIMAX/SARIMAX (calendar features from aggregate series)
            if self.ts_train is not None:
                n_train = len(self.ts_train)
                n_test = len(self.ts_test) if self.ts_test is not None else 0
                # Use simple time indices as exog
                exog_train = np.column_stack([
                    np.arange(n_train),
                    np.sin(2 * np.pi * np.arange(n_train) / 7),
                    np.cos(2 * np.pi * np.arange(n_train) / 7)
                ])
                exog_test = np.column_stack([
                    np.arange(n_train, n_train + n_test),
                    np.sin(2 * np.pi * np.arange(n_train, n_train + n_test) / 7),
                    np.cos(2 * np.pi * np.arange(n_train, n_train + n_test) / 7)
                ])
                self._exog_test_ts = exog_test
            
            try:
                naive = NaiveForecaster(seasonal=False)
                naive.fit(self.ts_train)
                self.models['Naive'] = naive
                self.logger.info("Naive model trained")
            except Exception as e:
                self.logger.error(f"Naive failed: {e}")

            try:
                seasonal_naive = NaiveForecaster(seasonal=True, seasonal_period=7)
                seasonal_naive.fit(self.ts_train)
                self.models['SeasonalNaive'] = seasonal_naive
                self.logger.info("Seasonal Naive model trained")
            except Exception as e:
                self.logger.error(f"Seasonal Naive failed: {e}")

            try:
                ets = ETSModel.auto_fit(self.ts_train, seasonal_periods=7)
                self.models['ETS'] = ets
                self.logger.info("ETS model trained")
            except Exception as e:
                self.logger.error(f"ETS failed: {e}")

            try:
                arimax = ARIMAXModel(auto_order=True, max_p=3, max_q=3)
                arimax.fit(self.ts_train, exog=exog_train)
                self.models['ARIMAX'] = arimax
                self.logger.info("ARIMAX model trained")
            except Exception as e:
                self.logger.error(f"ARIMAX failed: {e}")

            try:
                sarimax = SARIMAXModel(
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 7),
                    auto_order=False
                )
                sarimax.fit(self.ts_train, exog=exog_train)
                self.models['SARIMAX'] = sarimax
                self.logger.info("SARIMAX model trained")
            except Exception as e:
                self.logger.error(f"SARIMAX failed: {e}")
        else:
            self.logger.warning("Skipping time-series models (insufficient series length)")

        # Linear Regression (baseline)
        try:
            lr = LinearRegressionModel()
            lr.fit(self.X_train, self.y_train)
            self.models['LinearRegression'] = lr
            self.logger.info("Linear Regression trained")
        except Exception as e:
            self.logger.error(f"Linear Regression failed: {e}")
        
        return self.models
    
    def train_ml_models(self) -> Dict:
        """Train ML models for Case 3."""
        
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
            xgb.fit(self.X_train, self.y_train)
            self.models['XGBoost'] = xgb
            self.logger.info("XGBoost trained")
        except Exception as e:
            self.logger.error(f"XGBoost failed: {e}")

        # 3. CatBoost — uses RAW categorical features (unscaled)
        try:
            cat = CatBoostModel()
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
                X_t, X_v, y_t, y_v = train_test_split(
                    self.X_train, self.y_train,
                    test_size=0.2, random_state=RANDOM_SEED
                )
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
                shallow = ShallowANNModel(scale_inputs=False)
                shallow.fit(X_t, y_t, X_val=X_v, y_val=y_v, verbose=0)
                self.models['ShallowANN'] = shallow
                self.logger.info("Shallow ANN trained")
            except Exception as e:
                self.logger.error(f"Shallow ANN failed: {e}")
        
        return self.models
    
    def evaluate_models(self) -> pd.DataFrame:
        """Evaluate all models for Case 3."""
        results_list = []
        ts_models_no_exog = ['Naive', 'SeasonalNaive', 'ETS']
        ts_models_exog = ['ARIMAX', 'SARIMAX']

        for model_name, model in self.models.items():
            try:
                if model_name in ts_models_no_exog:
                    if self.ts_test is None or len(self.ts_test) == 0:
                        self.logger.warning(f"No time-series test data for {model_name}")
                        continue
                    preds = model.predict(len(self.ts_test))
                    self.predictions[model_name] = preds
                    metrics = self.metrics.evaluate_model(
                        model_name=model_name,
                        y_true=self.ts_test,
                        y_pred=preds,
                        case=self.case_name,
                        dataset_split='test'
                    )
                elif model_name in ts_models_exog:
                    if self.ts_test is None or len(self.ts_test) == 0:
                        self.logger.warning(f"No time-series test data for {model_name}")
                        continue
                    exog_future = getattr(self, '_exog_test_ts', None)
                    preds = model.predict(len(self.ts_test), exog_future=exog_future)
                    self.predictions[model_name] = preds
                    metrics = self.metrics.evaluate_model(
                        model_name=model_name,
                        y_true=self.ts_test,
                        y_pred=preds,
                        case=self.case_name,
                        dataset_split='test'
                    )
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
        """Generate figures for Case 3."""
        super().generate_figures()

        # Figure 6.3: Time series of Shipping Delivery Days
        try:
            date_col = 'order date (DateOrders)'
            if date_col in self.raw_data.columns:
                ts_data = self.raw_data.copy()
                ts_data[date_col] = pd.to_datetime(ts_data[date_col], errors='coerce')
                ts_data = ts_data.sort_values(date_col)
                # Aggregate by day for better visualization
                daily_agg = ts_data.groupby(ts_data[date_col].dt.date)[self.target_col].mean().reset_index()
                daily_agg.columns = ['Date', self.target_col]
                daily_agg['Date'] = pd.to_datetime(daily_agg['Date'])
                self.visualizer.plot_time_series(
                    data=daily_agg,
                    date_col='Date',
                    value_col=self.target_col,
                    title='Shipping Delivery Days Over Time',
                    filename='figure_6_3_case3_timeseries',
                    chapter='chapter6'
                )
        except Exception as e:
            self.logger.error(f"Figure 6.3 failed: {e}")

        # Figure 6.6: Distribution of Delivery Days
        try:
            self.visualizer.plot_distribution(
                data=self.processed_data[self.target_col].dropna(),
                title='Distribution of Shipping Delivery Days',
                filename='figure_6_6_case3_distribution',
                xlabel='Days for Shipping (Real)',
                chapter='chapter6'
            )
        except Exception as e:
            self.logger.error(f"Figure 6.6 failed: {e}")

        # Figure 6.8: Feature-target relationships for shipping delivery days
        try:
            features_to_plot = ['Days for shipment (scheduled)', 'Order Item Quantity',
                               'Order Item Total', 'Product Price']
            categorical_features = ['Shipping Mode', 'Order Region']
            all_features = [f for f in features_to_plot + categorical_features
                           if f in self.raw_data.columns]
            cat_features = [f for f in categorical_features if f in self.raw_data.columns]
            self.visualizer.plot_feature_target_relationships(
                data=self.raw_data,
                features=all_features[:6],  # Limit to 6 for clean layout
                target_col=self.target_col,
                categorical_features=cat_features,
                title='Feature-Target Relationships for Shipping Delivery Days',
                filename='figure_6_8_case3_feature_target',
                chapter='chapter6'
            )
        except Exception as e:
            self.logger.error(f"Figure 6.8 failed: {e}")
        
        # Figure 7.7: Actual vs Predicted (Random Forest)
        if 'RandomForest' in self.predictions:
            try:
                # Create subplot comparison
                self.visualizer.plot_actual_vs_predicted(
                    y_true=self.y_test,
                    y_pred=self.predictions['RandomForest'],
                    model_name='Random Forest',
                    title='Actual vs Predicted Delivery Days (Random Forest)',
                    filename='figure_7_7_case3_rf_scatter',
                    chapter='chapter7'
                )
            except Exception as e:
                self.logger.error(f"Figure 7.7 failed: {e}")

        # Figure 7.7: Actual vs Predicted (XGBoost)
        if 'XGBoost' in self.predictions:
            try:
                self.visualizer.plot_actual_vs_predicted(
                    y_true=self.y_test,
                    y_pred=self.predictions['XGBoost'],
                    model_name='XGBoost',
                    title='Actual vs Predicted Delivery Days (XGBoost)',
                    filename='figure_7_7_case3_xgb_scatter',
                    chapter='chapter7'
                )
            except Exception as e:
                self.logger.error(f"Figure 7.7 (XGBoost) failed: {e}")
        
        # Figure 7.8: Distribution of Prediction Errors (per PDF specification)
        # Per PDF: Use boxplots or density plots for classical vs intelligent models
        try:
            # Compute forecast errors (actual - predicted)
            errors = {
                model: self.y_test - preds
                for model, preds in self.predictions.items()
                if len(preds) == len(self.y_test)
            }
            # Plot KDE distribution per PDF Figure 7.8 specification
            self.visualizer.plot_error_distribution_kde(
                errors=errors,
                title='Distribution of Prediction Errors (Case 3 - Shipping)',
                filename='figure_7_8_case3_error_distribution',
                chapter='chapter7'
            )
        except Exception as e:
            self.logger.error(f"Figure 7.8 failed: {e}")
        
        # Feature Importance (Random Forest)
        if 'RandomForest' in self.models:
            try:
                importance_df = self.models['RandomForest'].get_feature_importance(top_n=15)
                self.visualizer.plot_feature_importance(
                    importance_df=importance_df,
                    title='Feature Importance for Shipping Delivery Days',
                    filename='figure_7_11_case3_importance',
                    chapter='chapter7'
                )
            except Exception as e:
                self.logger.error(f"Feature importance figure failed: {e}")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    config = Config()
    experiment = Case3Experiment(config)
    results = experiment.run()
    
    print("\nCase 3 Results:")
    print(results.to_string())
