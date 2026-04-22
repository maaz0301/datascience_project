import pytest
import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import FraudDetector
from config import MODELS_DIR

def create_dummy_assets():
    """Create dummy model and scalers for testing if they don't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "random_forest_model.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scalers.pkl")
    
    if not os.path.exists(model_path):
        X = np.random.randn(10, 31) # Time, V1-V28, Amount
        y = np.random.randint(0, 2, 10)
        model = RandomForestClassifier(n_estimators=1)
        model.fit(X, y)
        joblib.dump(model, model_path)
        
    if not os.path.exists(scaler_path):
        scalers = {
            "Amount": RobustScaler().fit(np.random.randn(10, 1)),
            "Time": RobustScaler().fit(np.random.randn(10, 1))
        }
        joblib.dump(scalers, scaler_path)

@pytest.fixture(scope="module")
def setup_detector():
    create_dummy_assets()
    return FraudDetector(model_name="random_forest")

def test_detector_initialization(setup_detector):
    """Test if detector loads models and scalers correctly."""
    assert setup_detector.model is not None
    assert setup_detector.scalers is not None
    assert "Amount" in setup_detector.scalers

def test_preprocess_record(setup_detector):
    """Test if preprocessing returns a DataFrame with expected dimensions."""
    raw_record = {
        "Time": 100.0,
        "Amount": 50.0,
        **{f"V{i}": np.random.randn() for i in range(1, 29)}
    }
    processed = setup_detector.preprocess_record(raw_record)
    assert isinstance(processed, pd.DataFrame)
    assert len(processed) == 1
    # Check if some engineered features are present
    assert "Hour" in processed.columns
    assert "PCA_Magnitude" in processed.columns

def test_prediction_output(setup_detector):
    """Test if predict returns expected dictionary format."""
    dummy_record = {
        "Time": 40000,
        "Amount": 100.0,
        **{f"V{i}": np.random.randn() for i in range(1, 29)}
    }
    result = setup_detector.predict(dummy_record)
    assert "prediction" in result
    assert "is_fraud" in result
    assert "probability" in result
    assert isinstance(result["is_fraud"], bool)
    assert 0 <= result["probability"] <= 1

def test_edge_case_zero_amount(setup_detector):
    """Test handling of zero amount transaction."""
    dummy_record = {
        "Time": 0,
        "Amount": 0.0,
        **{f"V{i}": 0.0 for i in range(1, 29)}
    }
    result = setup_detector.predict(dummy_record)
    assert result["prediction"] in ["Fraud", "Legitimate"]

def test_missing_v_features(setup_detector):
    """Test detector resilience to missing features (should handle gracefully if possible)."""
    # Note: inference.py's preprocess_record expects V1-V28
    # If we pass only Time and Amount, we might expect it to handle it if we add safety logic
    partial_record = {
        "Time": 100.0,
        "Amount": 50.0
    }
    # This might fail currently, which is good to know
    with pytest.raises(Exception):
        setup_detector.predict(partial_record)
