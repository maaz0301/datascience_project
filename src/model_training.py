"""
Model Training Module
=====================
Trains multiple machine learning models for fraud detection.
Implements SMOTE for handling class imbalance.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    RANDOM_STATE, MODELS_DIR, CV_FOLDS,
    SMOTE_SAMPLING_STRATEGY, SMOTE_K_NEIGHBORS,
    LOGISTIC_REGRESSION_PARAMS, RANDOM_FOREST_PARAMS,
    GRADIENT_BOOSTING_PARAMS
)


def apply_smote(X_train, y_train, sampling_strategy=SMOTE_SAMPLING_STRATEGY,
                k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_STATE):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance classes.
    
    SMOTE creates synthetic samples of the minority class by interpolating
    between existing minority samples, rather than simply duplicating them.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training labels.
    sampling_strategy : float
        Ratio of minority to majority class after resampling.
    k_neighbors : int
        Number of nearest neighbors for SMOTE.
    random_state : int
        Random seed.
    
    Returns
    -------
    tuple
        (X_resampled, y_resampled)
    """
    print(f"\n  ⚖️  APPLYING SMOTE (Synthetic Minority Over-sampling)")
    print(f"  {'─'*50}")
    
    # Before SMOTE
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"  Before SMOTE:")
    for label, count in zip(unique, counts):
        name = "Legitimate" if label == 0 else "Fraud"
        print(f"    {name}: {count:,}")
    print(f"    Ratio: {counts[0]/counts[1]:.0f}:1")
    
    # Apply SMOTE
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # After SMOTE
    unique, counts = np.unique(y_resampled, return_counts=True)
    print(f"\n  After SMOTE:")
    for label, count in zip(unique, counts):
        name = "Legitimate" if label == 0 else "Fraud"
        print(f"    {name}: {count:,}")
    print(f"    Ratio: {counts[0]/counts[1]:.1f}:1")
    print(f"    New total samples: {len(y_resampled):,}")
    
    return X_resampled, y_resampled


def get_models():
    """
    Get a dictionary of models to train.
    
    Returns
    -------
    dict
        {model_name: model_instance}
    """
    models = {
        "Logistic Regression": LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
        "Random Forest": RandomForestClassifier(**RANDOM_FOREST_PARAMS),
        "Gradient Boosting": GradientBoostingClassifier(**GRADIENT_BOOSTING_PARAMS),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        )
    }
    
    return models


def train_model(model, X_train, y_train, model_name="Model"):
    """
    Train a single model and return training details.
    
    Parameters
    ----------
    model : sklearn estimator
        Model to train.
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    model_name : str
        Name of the model for display.
    
    Returns
    -------
    dict
        Training results including trained model and timing.
    """
    print(f"\n  🔄 Training {model_name}...", end="", flush=True)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f" Done! ({training_time:.2f}s)")
    
    # Get predictions on training data
    train_accuracy = model.score(X_train, y_train)
    print(f"    Training accuracy: {train_accuracy:.4f}")
    
    return {
        "model": model,
        "training_time": training_time,
        "train_accuracy": train_accuracy
    }


def cross_validate_model(model, X_train, y_train, model_name="Model", cv=CV_FOLDS):
    """
    Perform stratified cross-validation.
    
    Parameters
    ----------
    model : sklearn estimator
        Model to cross-validate.
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    model_name : str
        Name of the model.
    cv : int
        Number of CV folds.
    
    Returns
    -------
    dict
        Cross-validation results.
    """
    print(f"\n  📊 Cross-Validating {model_name} ({cv}-Fold)...", end="", flush=True)
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    
    # Score with multiple metrics
    scoring_metrics = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }
    
    cv_results = {}
    
    for metric_name, scorer in scoring_metrics.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=skf, scoring=scorer, n_jobs=-1)
            cv_results[metric_name] = {
                "mean": scores.mean(),
                "std": scores.std(),
                "scores": scores
            }
        except Exception:
            cv_results[metric_name] = {"mean": 0, "std": 0, "scores": np.array([0]*cv)}
    
    print(f" Done!")
    print(f"    CV Results ({cv}-Fold):")
    for metric, result in cv_results.items():
        print(f"      {metric:>12}: {result['mean']:.4f} (+/- {result['std']:.4f})")
    
    return cv_results


def train_all_models(X_train, y_train, use_smote=True, primary_only=False):
    """
    Train models with optional SMOTE.
    
    Parameters
    ----------
    X_train : pd.DataFrame or np.ndarray
        Training features.
    y_train : pd.Series or np.ndarray
        Training labels.
    use_smote : bool
        Whether to apply SMOTE before training.
    primary_only : bool
        If True, only train the primary model (Random Forest) for speed.
    """
    print(f"\n{'='*60}")
    print(f"  🤖 MODEL TRAINING PIPELINE")
    print(f"{'='*60}")
    
    # Apply SMOTE if requested
    if use_smote:
        X_train_balanced, y_train_balanced = apply_smote(X_train, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
        print(f"\n  ⚠️  Training WITHOUT SMOTE (original imbalanced data)")
    
    # Get models
    models = get_models()
    
    if primary_only:
        print(f"\n  🚀 FAST MODE: Only training primary model (Random Forest)")
        models = {"Random Forest": models["Random Forest"]}
        
    trained_models = {}
    
    print(f"\n  📋 Models to train: {', '.join(models.keys())}")
    print(f"  {'─'*50}")
    
    for name, model in models.items():
        # Train model
        result = train_model(model, X_train_balanced, y_train_balanced, name)
        
        # Cross-validation (on original imbalanced data for realistic assessment)
        cv_results = cross_validate_model(model, X_train, y_train, name)
        
        result["cv_results"] = cv_results
        trained_models[name] = result
    
    # Summary
    print(f"\n{'='*60}")
    print(f"  📊 TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Model':<25} {'Train Time':>12} {'CV F1':>10} {'CV AUC':>10}")
    print(f"  {'─'*60}")
    
    for name, result in trained_models.items():
        cv_f1 = result["cv_results"].get("f1", {}).get("mean", 0)
        cv_auc = result["cv_results"].get("roc_auc", {}).get("mean", 0)
        print(f"  {name:<25} {result['training_time']:>10.2f}s {cv_f1:>10.4f} {cv_auc:>10.4f}")
    
    print(f"\n  ✅ All models trained successfully!")
    print(f"{'='*60}")
    
    return trained_models


def save_model(model, model_name, directory=MODELS_DIR):
    """Save a trained model to disk."""
    os.makedirs(directory, exist_ok=True)
    filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    filepath = os.path.join(directory, filename)
    joblib.dump(model, filepath)
    print(f"  💾 Saved: {filepath}")
    return filepath


def load_model(model_name, directory=MODELS_DIR):
    """Load a trained model from disk."""
    filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
    filepath = os.path.join(directory, filename)
    model = joblib.load(filepath)
    print(f"  📂 Loaded: {filepath}")
    return model


def save_all_models(trained_models, directory=MODELS_DIR):
    """Save all trained models."""
    print(f"\n  💾 Saving all models...")
    for name, result in trained_models.items():
        save_model(result["model"], name, directory)
    print(f"  ✅ All models saved to: {directory}")


if __name__ == "__main__":
    from data_loader import load_dataset
    from preprocessing import preprocess_pipeline
    
    df = load_dataset()
    X_train, X_test, y_train, y_test, scalers, _ = preprocess_pipeline(df)
    trained_models = train_all_models(X_train, y_train, use_smote=True)
    save_all_models(trained_models)
