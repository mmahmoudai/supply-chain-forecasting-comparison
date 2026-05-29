"""Case 1: Historical Product Demand Experiment."""

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
from models.classical import NaiveForecaster, ETSModel, ARIMAXModel, SARIMAXModel, LinearRegressionModel
from models.ml import XGBoostModel, CatBoostModel, FFNNModel, RandomForestModel, ShallowANNModel


class Case1Experiment(BaseExperiment):
    """
    Experiment for Case 1: Historical Product Demand.
    
    Models: Naive, ETS, ARIMA, Linear Regression, XGBoost, FFNN
    """
    
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.target_col = TARGET_COLUMNS['case1']
        self.date_col = DATE_COLUMNS['case1']
        self.aggregated_data = None
    
    @property
    def case_name(self) -> str:
        return 'case1'
    
    def load_data(self) -> pd.DataFrame:
        """Load Case 1 dataset."""
        self.raw_data = self.data_loader.load_case1()
        self.logger.info(f"Loaded {len(self.raw_data)} rows")
        return self.raw_data
    
    def preprocess(self) -> pd.DataFrame:
        """Preprocess Case 1 data."""
        self.processed_data = self.preprocessor.preprocess_case1(self.raw_data)
        return self.processed_data
    
    def engineer_features(self) -> pd.DataFrame:
        """Create features for Case 1."""
        # Create aggregated features for ML models
        self.aggregated_data = self.feature_engineer.engineer_case1_features(
            self.processed_data, 
            aggregate=True
        )
        self.logger.info(f"Feature engineered data: {len(self.aggregated_data)} rows")
        return self.aggregated_data
    
    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Time-based train/test split for Case 1."""
        df = self.aggregated_data.copy()
        df = df.sort_values(self.date_col)
        
        # Time-based split
        split_idx = int(len(df) * TRAIN_RATIO)
        
        self.train_data = df.iloc[:split_idx].copy()
        self.test_data = df.iloc[split_idx:].copy()
        
        # Prepare X and y for ML models
        feature_cols = self.feature_engineer.get_feature_names('case1')
        self.feature_cols = feature_cols
        
        self.X_train = self.train_data[feature_cols]
        self.X_test = self.test_data[feature_cols]
        self.y_train = self.train_data[self.target_col].values
        self.y_test = self.test_data[self.target_col].values
        
        self.logger.info(f"Train: {len(self.train_data)}, Test: {len(self.test_data)}")
        return self.train_data, self.test_data
    
    def train_classical_models(self) -> Dict:
        """Train classical statistical models for Case 1."""
        
        # Get univariate time series for classical models
        y_series = self.train_data[self.target_col].values
        
        # Build exogenous features for ARIMAX/SARIMAX (calendar features at aggregate level)
        exog_cols = [c for c in ['month', 'quarter', 'month_sin', 'month_cos',
                                 'quarter_sin', 'quarter_cos', 'time_index']
                     if c in self.train_data.columns]
        exog_train = self.train_data[exog_cols].values if exog_cols else None
        exog_test = self.test_data[exog_cols].values if exog_cols else None
        self._exog_test = exog_test  # Store for evaluate_models
        
        # 1. Naive Forecaster
        try:
            naive = NaiveForecaster(seasonal=False)
            naive.fit(y_series)
            self.models['Naive'] = naive
            self.logger.info("Naive model trained")
        except Exception as e:
            self.logger.error(f"Naive failed: {e}")

        # 1b. Seasonal Naive
        try:
            seasonal_naive = NaiveForecaster(seasonal=True, seasonal_period=7)
            seasonal_naive.fit(y_series)
            self.models['SeasonalNaive'] = seasonal_naive
            self.logger.info("Seasonal Naive model trained")
        except Exception as e:
            self.logger.error(f"Seasonal Naive failed: {e}")
        
        # 2. Exponential Smoothing (ETS)
        try:
            ets = ETSModel.auto_fit(y_series, seasonal_periods=7)
            self.models['ETS'] = ets
            self.logger.info("ETS model trained")
        except Exception as e:
            self.logger.error(f"ETS failed: {e}")
        
        # 3. ARIMAX
        try:
            arimax = ARIMAXModel(auto_order=True, max_p=3, max_q=3)
            arimax.fit(y_series, exog=exog_train)
            self.models['ARIMAX'] = arimax
            self.logger.info("ARIMAX model trained")
        except Exception as e:
            self.logger.error(f"ARIMAX failed: {e}")

        # 3b. SARIMAX
        try:
            sarimax = SARIMAXModel(
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                auto_order=False
            )
            sarimax.fit(y_series, exog=exog_train)
            self.models['SARIMAX'] = sarimax
            self.logger.info("SARIMAX model trained")
        except Exception as e:
            self.logger.error(f"SARIMAX failed: {e}")
        
        # 4. Linear Regression with lag features
        try:
            lr = LinearRegressionModel()
            lr.fit(self.X_train, self.y_train)
            self.models['LinearRegression'] = lr
            self.logger.info("Linear Regression trained")
        except Exception as e:
            self.logger.error(f"Linear Regression failed: {e}")
        
        return self.models
    
    def train_ml_models(self) -> Dict:
        """Train ML models for Case 1.

        Per Table 6.3: Linear Regression (lags), XGBoost, FFNN
        """

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

        # 3. CatBoost
        try:
            cat = CatBoostModel()
            cat.fit(self.X_train, self.y_train)
            self.models['CatBoost'] = cat
            self.logger.info("CatBoost trained")
        except Exception as e:
            self.logger.error(f"CatBoost failed: {e}")

        # 4. FFNN and Shallow ANN
        if self.config.run_neural_networks:
            try:
                # Split some training data for validation
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
                shallow = ShallowANNModel()
                shallow.fit(X_t, y_t, X_val=X_v, y_val=y_v, verbose=0)
                self.models['ShallowANN'] = shallow
                self.logger.info("Shallow ANN trained")
            except Exception as e:
                self.logger.error(f"Shallow ANN failed: {e}")
        
        return self.models
    
    def evaluate_models(self) -> pd.DataFrame:
        """Evaluate all models for Case 1."""
        results_list = []
        
        for model_name, model in self.models.items():
            try:
                # Generate predictions
                if model_name in ['Naive', 'SeasonalNaive', 'ETS']:
                    # Time series models - forecast (no exog)
                    preds = model.predict(len(self.y_test))
                elif model_name in ['ARIMAX', 'SARIMAX']:
                    # Time series models with exogenous variables
                    exog_future = getattr(self, '_exog_test', None)
                    preds = model.predict(len(self.y_test), exog_future=exog_future)
                else:
                    # Supervised models
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
        """Generate figures for Case 1."""
        super().generate_figures()
        
        # Figure 6.1: Time series plot
        try:
            self.visualizer.plot_time_series(
                data=self.aggregated_data,
                date_col=self.date_col,
                value_col=self.target_col,
                title='Historical Product Demand Over Time',
                filename='figure_6_1_case1_timeseries',
                chapter='chapter6'
            )
        except Exception as e:
            self.logger.error(f"Figure 6.1 failed: {e}")
        
        # Figure 6.4: Distribution plot
        try:
            self.visualizer.plot_distribution(
                data=self.aggregated_data[self.target_col],
                title='Distribution of Order Demand',
                filename='figure_6_4_case1_distribution',
                xlabel='Order Demand',
                chapter='chapter6'
            )
        except Exception as e:
            self.logger.error(f"Figure 6.4 failed: {e}")

        # Figure 6.7: Lagged demand vs future demand relationship
        if 'lag_1' in self.aggregated_data.columns:
            try:
                self.visualizer.plot_lagged_scatter(
                    data=self.aggregated_data,
                    lag_col='lag_1',
                    target_col=self.target_col,
                    title='Lagged Demand vs Future Demand Relationship',
                    filename='figure_6_7_lagged_demand',
                    chapter='chapter6',
                    add_regression=True
                )
            except Exception as e:
                self.logger.error(f"Figure 6.7 failed: {e}")
        
        # Figure 7.1: Actual vs Predicted (ARIMA vs ML models)
        if 'ARIMA' in self.predictions:
            try:
                model_preds = {'ARIMA': self.predictions['ARIMA']}
                ml_models = []

                if 'XGBoost' in self.predictions:
                    model_preds['XGBoost'] = self.predictions['XGBoost']
                    ml_models.append('XGBoost')
                if 'FFNN' in self.predictions:
                    model_preds['FFNN'] = self.predictions['FFNN']
                    ml_models.append('FFNN')

                if ml_models:
                    title = f"Actual vs Predicted: ARIMA vs {' vs '.join(ml_models)} (Case 1)"
                else:
                    title = 'Actual vs Predicted: ARIMA (Case 1)'

                self.visualizer.plot_model_comparison_line(
                    data=self.test_data,
                    date_col=self.date_col,
                    actual_col=self.target_col,
                    predictions=model_preds,
                    title=title,
                    filename='figure_7_1_case1_comparison',
                    chapter='chapter7'
                )
            except Exception as e:
                self.logger.error(f"Figure 7.1 failed: {e}")
        
        # Figure 7.2: Distribution of Forecast Errors (histograms/KDE per PDF)
        # Per PDF: Compare ARIMA vs FFNN (classical vs intelligent model)
        if 'ARIMA' in self.predictions:
            try:
                # Compute forecast errors (actual - predicted) per PDF spec
                errors = {
                    model: self.y_test - preds
                    for model, preds in self.predictions.items()
                    if len(preds) == len(self.y_test)
                }
                if errors:
                    # Plot KDE distribution per PDF Figure 7.2 specification
                    self.visualizer.plot_error_distribution_kde(
                        errors=errors,
                        title='Distribution of Forecast Errors (Case 1)',
                        filename='figure_7_2_case1_error_distribution',
                        chapter='chapter7'
                    )
            except Exception as e:
                self.logger.error(f"Figure 7.2 failed: {e}")

        # Figure 7.9: Residual time series (ARIMA vs ML models)
        if 'ARIMA' in self.predictions:
            try:
                residuals = {}
                models_to_plot = ['ARIMA']
                for model_name in ['XGBoost', 'FFNN']:
                    if model_name in self.predictions:
                        models_to_plot.append(model_name)

                for model_name in models_to_plot:
                    if model_name in self.predictions:
                        preds = self.predictions[model_name]
                        if len(preds) == len(self.y_test):
                            residuals[model_name] = self.y_test - preds
                if residuals:
                    time_idx = np.arange(len(self.y_test))
                    self.visualizer.plot_residual_time_series(
                        time_index=time_idx,
                        residuals_dict=residuals,
                        title='Residual Time Series (Case 1)',
                        filename='figure_7_9_case1_residuals',
                        chapter='chapter7'
                    )
            except Exception as e:
                self.logger.error(f"Figure 7.9 failed: {e}")

        # Figure 7.13: Accuracy vs Dataset Size (using Case 1 as the largest dataset)
        try:
            model_classes = {
                'LinearRegression': LinearRegressionModel,
                'RandomForest': RandomForestModel,
                'XGBoost': XGBoostModel
            }
            self.generate_accuracy_vs_size_figure(
                model_classes=model_classes,
                sizes=[0.25, 0.5, 0.75, 1.0],
                metric='RMSE'
            )
        except Exception as e:
            self.logger.error(f"Figure 7.13 failed: {e}")

        # Figure 7.11: Feature Importance (XGBoost)
        if 'XGBoost' in self.models:
            try:
                importance_df = self.models['XGBoost'].get_feature_importance(
                    top_n=15,
                    feature_names=getattr(self, 'feature_cols', None)
                )
                self.visualizer.plot_feature_importance(
                    importance_df=importance_df,
                    title='Feature Importance for Product Demand (XGBoost)',
                    filename='figure_7_11_case1_importance',
                    chapter='chapter7'
                )
            except Exception as e:
                self.logger.error(f"Feature importance figure failed: {e}")


if __name__ == "__main__":
    # Run Case 1 experiment standalone
    import logging
    logging.basicConfig(level=logging.INFO)
    
    config = Config()
    experiment = Case1Experiment(config)
    results = experiment.run()
    
    print("\nCase 1 Results:")
    print(results.to_string())
