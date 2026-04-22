"""
Configuration settings for the Credit Card Fraud Detection project.
Centralized configuration for paths, model parameters, and visualization settings.
"""

import os

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")

# Dataset
DATASET_PATH = os.path.join(DATA_DIR, "creditcard.csv")

# Create directories if they don't exist
for directory in [DATA_DIR, REPORTS_DIR, MODELS_DIR, FIGURES_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================
TARGET_COLUMN = "Class"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================
SYNTHETIC_NUM_TRANSACTIONS = 284807  # Same size as Kaggle dataset
SYNTHETIC_FRAUD_RATIO = 0.00173     # ~0.17% fraud rate (realistic)

# ============================================================================
# PREPROCESSING
# ============================================================================
SCALING_COLUMNS = ["Amount", "Time"]
PCA_FEATURES = [f"V{i}" for i in range(1, 29)]

# ============================================================================
# MODEL TRAINING
# ============================================================================
# SMOTE parameters
SMOTE_SAMPLING_STRATEGY = 0.5  # Minority will be 50% of majority
SMOTE_K_NEIGHBORS = 5

# Cross-validation
CV_FOLDS = 5

# Model hyperparameters
LOGISTIC_REGRESSION_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "solver": "lbfgs"
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "random_state": RANDOM_STATE
}

XGBOOST_STYLE_PARAMS = {
    "n_estimators": 150,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "random_state": RANDOM_STATE
}

# ============================================================================
# EVALUATION THRESHOLDS
# ============================================================================
FRAUD_THRESHOLD = 0.5
OPTIMAL_THRESHOLD_METRIC = "f1"  # Optimize threshold for F1 score

# ============================================================================
# VISUALIZATION
# ============================================================================
FIGURE_DPI = 150
FIGURE_STYLE = "seaborn-v0_8-darkgrid"
COLOR_PALETTE = {
    "legitimate": "#2ecc71",
    "fraud": "#e74c3c",
    "primary": "#3498db",
    "secondary": "#9b59b6",
    "accent": "#f39c12",
    "dark": "#2c3e50",
    "light": "#ecf0f1"
}

PLOT_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
