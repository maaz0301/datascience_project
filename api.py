from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from src.inference import FraudDetector
import os
import uvicorn

app = FastAPI(
    title="Guardian Sentinel API",
    description="REST API for real-time credit card fraud detection.",
    version="1.0.0"
)

# Initialize detector
try:
    detector = FraudDetector(model_name="random_forest")
except Exception as e:
    # In a real production app, we might want to log this and handle it better
    detector = None
    print(f"Warning: Could not initialize model: {e}")

class TransactionData(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

@app.get("/health")
def health_check():
    return {
        "status": "online" if detector else "degraded",
        "model_loaded": detector is not None,
        "api_version": "1.0.0"
    }

@app.post("/predict")
def predict_fraud(data: TransactionData):
    if not detector:
        raise HTTPException(status_code=503, detail="Fraud detection model not loaded.")
    
    try:
        # Convert Pydantic model to dict
        record = data.dict()
        result = detector.predict(record)
        return {
            "is_fraud": result["is_fraud"],
            "prediction": result["prediction"],
            "probability": result["probability"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
