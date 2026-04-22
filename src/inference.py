"""
Inference Module
================
Handles real-time fraud detection using trained models and saved scalers.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODELS_DIR, SCALING_COLUMNS
from src.feature_engineering import create_time_features, create_amount_features, create_pca_interaction_features

class FraudDetector:
    """
    Main class for real-time fraud detection.
    Loads models and scalers to perform end-to-end inference.
    """
    
    def __init__(self, model_name="random_forest"):
        """
        Initialize the detector by loading models and scalers.
        """
        self.model_path = os.path.join(MODELS_DIR, f"{model_name}_model.pkl")
        self.scalers_path = os.path.join(MODELS_DIR, "scalers.pkl")
        
        # Load components
        try:
            self.model = joblib.load(self.model_path)
            self.scalers = joblib.load(self.scalers_path)
            print(f"[OK] Loaded model: {model_name}")
            print(f"[OK] Loaded {len(self.scalers)} scalers")
        except Exception as e:
            raise FileNotFoundError(f"Could not load detection components: {e}. "
                                    f"Run training first to generate .pkl files.")

    def preprocess_record(self, raw_record):
        """
        Apply the full feature engineering and scaling pipeline to a single record.
        
        Parameters
        ----------
        raw_record : dict or pd.Series
            Raw transaction data containing 'Time', 'Amount', and 'V1'-'V28'.
            
        Returns
        -------
        pd.DataFrame
            Fully processed record ready for prediction.
        """
        if isinstance(raw_record, dict):
            df = pd.DataFrame([raw_record])
        else:
            df = raw_record.to_frame().T
            
        # 0. Handle initial inf/nan before engineering
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
            
        # 1. Feature Engineering (Match training pipeline)
        df = create_time_features(df, verbose=False)
        df = create_amount_features(df, verbose=False)
        df = create_pca_interaction_features(df, verbose=False)
        
        # 2. Scaling
        for col, scaler in self.scalers.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]])
        
        return df

    def predict(self, raw_record):
        """
        Predict if a transaction is fraudulent.
        
        Returns
        -------
        dict
            Prediction results including label and probability.
        """
        # Preprocess
        processed_df = self.preprocess_record(raw_record)
        
        # Ensure feature order matches model
        if hasattr(self.model, "feature_names_in_"):
            processed_df = processed_df[self.model.feature_names_in_]
            
        # Predict
        prediction = self.model.predict(processed_df)[0]
        probability = self.model.predict_proba(processed_df)[0][1]
        
        return {
            "prediction": "Fraud" if prediction == 1 else "Legitimate",
            "is_fraud": bool(prediction == 1),
            "probability": float(probability),
            "processed_data": processed_df.iloc[0].to_dict()
        }

    def explain_prediction(self, raw_record):
        """
        Explain why a prediction was made by calculating feature contributions.
        
        Returns
        -------
        pd.DataFrame
            Feature impact scores sorted by magnitude.
        """
        processed_df = self.preprocess_record(raw_record)
        
        if hasattr(self.model, "feature_names_in_"):
            processed_df = processed_df[self.model.feature_names_in_]
            
        feature_names = processed_df.columns.tolist()
        feature_values = processed_df.iloc[0].values
        
        # Calculate impact
        # For a truly accurate explanation, we'd use SHAP, but for real-time 
        # UI performance, we'll use a combination of feature importance 
        # and deviations from typical values.
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            # Simplified impact: Importance * abs(normalized_value)
            # This identifies high-value features that the model weights heavily
            impacts = importances * np.abs(feature_values)
            
            explanations = pd.DataFrame({
                'feature': feature_names,
                'value': feature_values,
                'impact': impacts
            })
            
            # Sort by impact
            explanations = explanations.sort_values(by='impact', ascending=False)
            return explanations
        
        return None

if __name__ == "__main__":
    # Test with a dummy record
    detector = FraudDetector()
    dummy_record = {
        "Time": 40000,
        "Amount": 100.0,
        **{f"V{i}": np.random.randn() for i in range(1, 29)}
    }
    
    result = detector.predict(dummy_record)
    print(f"\nTest Prediction Result:")
    print(f"Result: {result['prediction']}")
    print(f"Fraud Probability: {result['probability']:.4f}")
