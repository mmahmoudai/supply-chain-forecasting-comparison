"""Constants and paths for the 4-Case Regression Analysis project."""

from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "new_dataset"  # Local dataset folder (for sharing with teammates)

# Dataset paths
CASE1_PATH = DATA_ROOT / "case 1" / "case 1-Historical Product Demand.csv"
CASE2_TRAIN_PATH = DATA_ROOT / "case 2" / "new_22032026" / "case_2_food_Demand_train_v3.csv"
CASE2_TEST_PATH = DATA_ROOT / "case 2" / "new_22032026" / "case_2_food_Demand_train_v3.csv"  # No separate test file
CASE3_PATH = DATA_ROOT / "case 3" / "DataCoSupplyChainDataset.csv"
CASE4_PATH = DATA_ROOT / "case 4" / "retail_store_inventory.csv"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
RESULTS_DIR = OUTPUTS_DIR / "results"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Random seed for reproducibility
RANDOM_SEED = 42

# Train/Test split ratio
TRAIN_RATIO = 0.8
TEST_RATIO = 0.2

# Case 2 specific splits (week-based)
CASE2_TRAIN_WEEKS = (1, 100)
CASE2_VAL_WEEKS = (101, 125)
CASE2_TEST_WEEKS = (126, 145)
CASE2_PRED_WEEKS = (146, 155)

# Figure settings
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
FIGURE_STYLE = "seaborn-v0_8-whitegrid"
DEFAULT_FIGSIZE = (10, 6)

# Model hyperparameters defaults
XGBOOST_DEFAULTS = {
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_SEED
}

CATBOOST_DEFAULTS = {
    'iterations': 200,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': RANDOM_SEED,
    'verbose': 0
}

RANDOM_FOREST_DEFAULTS = {
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

FFNN_DEFAULTS = {
    'hidden_layers': [128, 64, 32],
    'activation': 'relu',
    'learning_rate': 0.001,
    'batch_size': 1024,
    'epochs': 200,
    'patience': 20
}

# Shallow ANN defaults (single hidden layer)
SHALLOW_ANN_DEFAULTS = {
    'hidden_units': 96,
    'activation': 'relu',
    'learning_rate': 0.0005,
    'batch_size': 1024,
    'epochs': 150,
    'patience': 15,
    'dropout_rate': 0.1,
    'scale_inputs': True,
    'scale_target': True
}

# Lag features configuration
LAG_FEATURES = {
    'case1': [1, 7, 14],
    'case2': [1, 4, 8],
    'case3': [],  # Cross-sectional, no lags
    'case4': [1, 7]
}

# Rolling window sizes
ROLLING_WINDOWS = {
    'case1': [7, 14],
    'case2': [4],
    'case4': [7]
}

# Target columns
TARGET_COLUMNS = {
    'case1': 'Order_Demand',
    'case2': 'num_orders',
    'case3': 'Days for shipping (real)',
    'case4': 'Units Sold'
}

# Date columns
DATE_COLUMNS = {
    'case1': 'Date',
    'case2': 'week',
    'case3': 'order date (DateOrders)',
    'case4': 'Date'
}

# Dataset 1 exact columns (Historical Product Demand)
CASE1_COLUMNS = ['Product_Code', 'Warehouse', 'Product_Category', 'Date', 'Order_Demand']

# Dataset 2 exact columns (Food Demand)
CASE2_COLUMNS = ['week', 'checkout_price', 'base_price', 'emailer_for_promotion',
                 'homepage_featured', 'Region', 'center_type', 'category', 'cuisine', 'num_orders']

# Dataset 4 exact columns (Retail Store Inventory)
CASE4_COLUMNS = ['Date', 'Store ID', 'Product ID', 'Category', 'Region', 'Inventory Level',
                 'Units Sold', 'Units Ordered', 'Demand Forecast', 'Price', 'Discount',
                 'Weather Condition', 'Holiday/Promotion', 'Competitor Pricing', 'Seasonality']

# Case 3 approved columns (leakage-free) - per PDF specification
CASE3_APPROVED_COLS = [
    # Shipment Planning Features
    'Days for shipment (scheduled)',
    'Shipping Mode',
    'Type',
    # Order & Commercial Characteristics
    'Order Item Quantity',
    'Order Item Total',
    'Sales',
    'Order Item Discount Rate',
    'Order Item Discount',
    'Product Price',
    'Latitude',
    'Longitude',
    # Geographic & Market Features
    'Market',
    'Order Region',
    'Order Country',
    'Order State',
    # For temporal feature extraction
    'order date (DateOrders)'
]

# Case 3 optional approved columns (use cautiously)
CASE3_OPTIONAL_COLS = [
    'Department Name',
    'Category Name',
    'Customer Segment'
]

# Case 3 excluded columns (data leakage / invalid)
CASE3_EXCLUDED_COLS = [
    # Outcome Leakage Variables
    'Delivery Status',
    'Late_delivery_risk',
    'shipping date (DateOrders)',
    # Post-Outcome Financial Variables
    'Order Profit Per Order',
    'Order Item Profit Ratio',
    # Identifiers & Personal Data
    'Order Id',
    'Order Item Id',
    'Customer Id',
    'Customer Email',
    'Customer Password',
    'Customer Fname',
    'Customer Lname',
    'Customer Street',
    'Customer Zipcode',
    # Product Descriptors Without Predictive Value
    'Product Image',
    'Product Description',
    'Product Name',
    'Product Status'
]

# Case 3 encoding rules
CASE3_ONEHOT_COLS = [
    'Shipping Mode',
    'Type',
    'Market',
    'Order Region',
    'Department Name',
    'Category Name',
    'Customer Segment'
]
CASE3_TARGET_ENCODE_COLS = ['Order Country', 'Order State']
CASE3_NUMERICAL_COLS = [
    'Order Item Quantity',
    'Order Item Total',
    'Sales',
    'Order Item Discount Rate',
    'Order Item Discount',
    'Product Price',
    'Latitude',
    'Longitude',
    'Days for shipment (scheduled)'
]

# Winsorization percentiles
WINSOR_LOWER = 1
WINSOR_UPPER = 99

# Categorical encoding thresholds
HIGH_CARDINALITY_THRESHOLD = 20
