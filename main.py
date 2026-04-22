"""
╔══════════════════════════════════════════════════════════════════════╗
║       CREDIT CARD FRAUD DETECTION - MAIN PIPELINE                  ║
║                                                                      ║
║  Real-Time Credit Card Fraud Detection System                        ║
║  Using Advanced Data Science Techniques                              ║
║                                                                      ║
║  Team: Allah Yar Durrani | Muskan Khan | M. Abdullah | Mujtaba Khan  ║
╚══════════════════════════════════════════════════════════════════════╝

This script orchestrates the complete end-to-end ML pipeline:
  1. Data Loading & Exploration
  2. Data Preprocessing (Cleaning, Scaling, Splitting)
  3. Feature Engineering (Time, Amount, PCA interactions)
  4. Exploratory Data Analysis Visualizations
  5. Model Training with SMOTE
  6. Model Evaluation & Comparison
  7. Visualization Generation
  8. Report Generation

Usage:
    python main.py              # Run full pipeline
    python main.py --fast       # Fast mode (one model)
    python main.py --help       # Show help

Prerequisites:
    pip install -r requirements.txt
    python generate_dataset.py  # Generate synthetic data (or use Kaggle dataset)
"""

import os
import sys
import time
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from config import (
    DATASET_PATH, FIGURES_DIR, REPORTS_DIR, MODELS_DIR,
    TARGET_COLUMN, RANDOM_STATE
)
from src.data_loader import load_dataset, explore_dataset
from src.preprocessing import preprocess_pipeline
from src.feature_engineering import feature_engineering_pipeline
from src.visualization import (
    generate_all_eda_plots,
    plot_confusion_matrix, plot_roc_curves,
    plot_precision_recall_curves, plot_model_comparison,
    plot_feature_importance, plot_threshold_analysis
)
from src.model_training import train_all_models, save_all_models, save_model
from src.evaluation import (
    evaluate_all_models, generate_report,
    print_why_accuracy_fails
)
import joblib


def print_banner():
    """Print the project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   🔒  CREDIT CARD FRAUD DETECTION SYSTEM  🔒                ║
    ║                                                              ║
    ║   Real-Time Detection Using Machine Learning                 ║
    ║   Advanced Data Science Techniques                           ║
    ║                                                              ║
    ║   Team Members:                                              ║
    ║     ➤ Allah Yar Durrani                                      ║
    ║     ➤ Muskan Khan                                            ║
    ║     ➤ M. Abdullah                                            ║
    ║     ➤ Mujtaba Khan                                           ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_pipeline(skip_eda=False, fast_mode=False):
    """
    Execute the complete fraud detection pipeline.
    
    Parameters
    ----------
    skip_eda : bool
        If True, skip EDA visualization generation.
    fast_mode : bool
        If True, only train the primary model for faster results.
    """
    pipeline_start = time.time()
    
    print_banner()
    
    # ================================================================
    # STEP 1: DATA LOADING
    # ================================================================
    print(f"\n{'▓'*60}")
    print(f"  STEP 1/7: DATA LOADING")
    print(f"{'▓'*60}")
    
    # Check if dataset exists, if not generate synthetic data
    if not os.path.exists(DATASET_PATH):
        print(f"\n  ⚠️  Dataset not found. Generating synthetic data...")
        from generate_dataset import generate_synthetic_dataset, save_dataset, print_dataset_summary
        df = generate_synthetic_dataset()
        print_dataset_summary(df)
        save_dataset(df)
        print(f"  ✅ Synthetic dataset created!")
    
    df = load_dataset()
    
    # ================================================================
    # STEP 2: EXPLORATORY DATA ANALYSIS
    # ================================================================
    print(f"\n{'▓'*60}")
    print(f"  STEP 2/7: EXPLORATORY DATA ANALYSIS")
    print(f"{'▓'*60}")
    
    eda_results = explore_dataset(df)
    
    # ================================================================
    # STEP 3: EDA VISUALIZATIONS
    # ================================================================
    if not skip_eda:
        print(f"\n{'▓'*60}")
        print(f"  STEP 3/7: EDA VISUALIZATIONS")
        print(f"{'▓'*60}")
        
        eda_plots = generate_all_eda_plots(df)
    else:
        print(f"\n  ⏭️  Skipping EDA visualizations (--skip-eda)")
    
    # ================================================================
    # STEP 4: FEATURE ENGINEERING
    # ================================================================
    print(f"\n{'▓'*60}")
    print(f"  STEP 4/7: FEATURE ENGINEERING")
    print(f"{'▓'*60}")
    
    df_engineered = feature_engineering_pipeline(df)
    
    # ================================================================
    # STEP 5: DATA PREPROCESSING
    # ================================================================
    print(f"\n{'▓'*60}")
    print(f"  STEP 5/7: DATA PREPROCESSING")
    print(f"{'▓'*60}")
    
    X_train, X_test, y_train, y_test, scalers, df_preprocessed = preprocess_pipeline(df_engineered)
    
    # Educational: Why accuracy fails
    print_why_accuracy_fails(y_test)
    
    # ================================================================
    # STEP 6: MODEL TRAINING
    # ================================================================
    print(f"\n{'▓'*60}")
    print(f"  STEP 6/7: MODEL TRAINING (with SMOTE)")
    if fast_mode:
        print(f"  🚀 RUNNING IN FAST MODE")
    print(f"{'▓'*60}")
    
    trained_models = train_all_models(X_train, y_train, use_smote=True, primary_only=fast_mode)
    
    # Save scalers for inference
    print(f"\n  📏 Saving scalers for real-time inference...")
    joblib.dump(scalers, os.path.join(MODELS_DIR, "scalers.pkl"))
    
    # Save models
    save_all_models(trained_models)
    
    # ================================================================
    # STEP 7: MODEL EVALUATION
    # ================================================================
    print(f"\n{'▓'*60}")
    print(f"  STEP 7/7: MODEL EVALUATION & VISUALIZATION")
    print(f"{'▓'*60}")
    
    # Evaluate all models
    all_results = evaluate_all_models(trained_models, X_test, y_test)
    
    # ---- Generate Evaluation Visualizations ----
    print(f"\n  📊 Generating evaluation visualizations...")
    
    # Confusion matrices for each model
    for name, results in all_results.items():
        plot_confusion_matrix(y_test, results["y_pred"], name)
    
    # ROC Curves comparison
    plot_roc_curves(all_results, y_test)
    
    # Precision-Recall Curves comparison
    plot_precision_recall_curves(all_results, y_test)
    
    # Model comparison dashboard
    plot_model_comparison(all_results)
    
    # Feature importance (for tree-based models)
    for name in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
        if name in trained_models:
            model = trained_models[name]["model"]
            if hasattr(model, "feature_importances_"):
                feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else \
                    [f"Feature_{i}" for i in range(X_train.shape[1])]
                plot_feature_importance(
                    model.feature_importances_,
                    feature_names,
                    name
                )
    
    # Threshold analysis for best model
    best_model_name = max(all_results.keys(), 
                          key=lambda k: all_results[k]["metrics"]["f1_score"])
    if all_results[best_model_name]["y_pred_proba"] is not None:
        plot_threshold_analysis(
            y_test.values if hasattr(y_test, 'values') else y_test,
            all_results[best_model_name]["y_pred_proba"],
            best_model_name
        )
    
    # ---- Generate Report ----
    print(f"\n  📄 Generating evaluation report...")
    report_path = generate_report(all_results)
    
    # ================================================================
    # PIPELINE COMPLETE
    # ================================================================
    pipeline_time = time.time() - pipeline_start
    
    print(f"\n{'═'*60}")
    print(f"  ✅ PIPELINE COMPLETE!")
    print(f"{'═'*60}")
    print(f"\n  ⏱️  Total execution time: {pipeline_time:.1f} seconds")
    print(f"\n  📁 Output Directories:")
    print(f"     Models  : {MODELS_DIR}")
    print(f"     Figures : {FIGURES_DIR}")
    print(f"     Reports : {REPORTS_DIR}")
    
    # Count generated files
    n_figures = len([f for f in os.listdir(FIGURES_DIR) if f.endswith('.png')]) if os.path.exists(FIGURES_DIR) else 0
    n_models = len([f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]) if os.path.exists(MODELS_DIR) else 0
    
    print(f"\n  📊 Generated: {n_figures} visualization figures")
    print(f"  💾 Saved: {n_models} trained models")
    print(f"  📄 Report: {os.path.join(REPORTS_DIR, 'evaluation_report.txt')}")
    
    # Print best model summary
    best_m = all_results[best_model_name]["metrics"]
    print(f"\n  🏆 BEST MODEL: {best_model_name}")
    print(f"     F1-Score  : {best_m['f1_score']:.4f}")
    print(f"     Precision : {best_m['precision']:.4f}")
    print(f"     Recall    : {best_m['recall']:.4f}")
    print(f"     ROC-AUC   : {best_m['roc_auc']:.4f}")
    
    print(f"\n{'═'*60}")
    print(f"  🎓 Credit Card Fraud Detection Project - Complete!")
    print(f"  📧 Team: Durrani, Khan, Abdullah, Khan")
    print(f"{'═'*60}\n")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Credit Card Fraud Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              Run the full pipeline
  python main.py --skip-eda   Skip EDA visualization generation
        """
    )
    parser.add_argument("--skip-eda", action="store_true",
                        help="Skip EDA visualization generation")
    parser.add_argument("--fast", action="store_true",
                        help="Only train primary model for faster results")
    
    args = parser.parse_args()
    
    results = run_pipeline(skip_eda=args.skip_eda, fast_mode=args.fast)
