"""
Model Evaluation Module
=======================
Comprehensive evaluation of trained fraud detection models.
Generates detailed metrics, comparison reports, and visualizations.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    average_precision_score, matthews_corrcoef, cohen_kappa_score
)
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REPORTS_DIR, FIGURES_DIR, FRAUD_THRESHOLD


def evaluate_model(model, X_test, y_test, model_name="Model", threshold=FRAUD_THRESHOLD):
    """
    Evaluate a single model with comprehensive metrics.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    X_test : array-like
        Test features.
    y_test : array-like
        True test labels.
    model_name : str
        Name of the model.
    threshold : float
        Classification threshold.
    
    Returns
    -------
    dict
        Comprehensive evaluation results.
    """
    print(f"\n  📏 Evaluating {model_name}...")
    print(f"  {'─'*55}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Get probability predictions (if available)
    y_pred_proba = None
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        # Apply custom threshold if different from default
        if threshold != 0.5:
            y_pred = (y_pred_proba >= threshold).astype(int)
    elif hasattr(model, "decision_function"):
        y_pred_proba = model.decision_function(X_test)
    
    # ---- Core Metrics ----
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    # ---- Advanced Metrics ----
    roc_auc = 0.0
    avg_precision = 0.0
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
    
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # ---- Derived Metrics ----
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0   # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0   # False Negative Rate
    
    # ---- Business Metrics ----
    # Assuming avg fraud amount = $150 and investigation cost = $10
    avg_fraud_amount = 150
    investigation_cost = 10
    money_saved = tp * avg_fraud_amount  # Frauds caught
    money_lost = fn * avg_fraud_amount   # Frauds missed
    investigation_waste = fp * investigation_cost  # False alarms cost
    net_benefit = money_saved - investigation_waste
    
    # ---- Print Results ----
    print(f"\n  {'METRIC':<30} {'VALUE':>10}")
    print(f"  {'─'*42}")
    print(f"  {'Accuracy':<30} {accuracy:>10.4f}")
    print(f"  {'Precision':<30} {precision:>10.4f}")
    print(f"  {'Recall (Sensitivity)':<30} {recall:>10.4f}")
    print(f"  {'Specificity':<30} {specificity:>10.4f}")
    print(f"  {'F1-Score':<30} {f1:>10.4f}")
    print(f"  {'ROC-AUC':<30} {roc_auc:>10.4f}")
    print(f"  {'Average Precision (PR-AUC)':<30} {avg_precision:>10.4f}")
    print(f"  {'Matthews Correlation Coeff.':<30} {mcc:>10.4f}")
    print(f"  {'Cohen\'s Kappa':<30} {kappa:>10.4f}")
    
    print(f"\n  CONFUSION MATRIX:")
    print(f"  {'─'*42}")
    print(f"  {'':>20} {'Predicted':^20}")
    print(f"  {'':>20} {'Legit':>10} {'Fraud':>10}")
    print(f"  {'Actual Legit':<20} {tn:>10,} {fp:>10,}")
    print(f"  {'Actual Fraud':<20} {fn:>10,} {tp:>10,}")
    
    print(f"\n  FALSE POSITIVE RATE: {fpr:.4f} ({fp:,} legitimate flagged as fraud)")
    print(f"  FALSE NEGATIVE RATE: {fnr:.4f} ({fn:,} frauds missed)")
    
    print(f"\n  💰 BUSINESS IMPACT (Estimated):")
    print(f"  {'─'*42}")
    print(f"  {'Fraud caught (savings)':<30} ${money_saved:>10,.0f}")
    print(f"  {'Fraud missed (losses)':<30} ${money_lost:>10,.0f}")
    print(f"  {'False alarm costs':<30} ${investigation_waste:>10,.0f}")
    print(f"  {'Net benefit':<30} ${net_benefit:>10,.0f}")
    
    # Build results dictionary
    results = {
        "model_name": model_name,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "avg_precision": avg_precision,
            "mcc": mcc,
            "kappa": kappa,
            "fpr": fpr,
            "fnr": fnr
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        },
        "business_impact": {
            "money_saved": money_saved,
            "money_lost": money_lost,
            "investigation_waste": investigation_waste,
            "net_benefit": net_benefit
        }
    }
    
    return results


def evaluate_all_models(trained_models, X_test, y_test):
    """
    Evaluate all trained models and generate comparison.
    
    Parameters
    ----------
    trained_models : dict
        {model_name: {"model": trained_model, ...}}
    X_test : array-like
        Test features.
    y_test : array-like
        True test labels.
    
    Returns
    -------
    dict
        {model_name: evaluation_results}
    """
    print(f"\n{'='*60}")
    print(f"  📊 MODEL EVALUATION - ALL MODELS")
    print(f"{'='*60}")
    
    all_results = {}
    
    for name, model_info in trained_models.items():
        model = model_info["model"]
        results = evaluate_model(model, X_test, y_test, name)
        results["training_time"] = model_info.get("training_time", 0)
        all_results[name] = results
    
    # ---- Comparison Table ----
    print(f"\n{'='*60}")
    print(f"  📋 MODEL COMPARISON TABLE")
    print(f"{'='*60}")
    
    header = f"  {'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'AUC':>8}"
    print(header)
    print(f"  {'─'*70}")
    
    for name, results in all_results.items():
        m = results["metrics"]
        print(f"  {name:<25} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>8.4f} {m['f1_score']:>8.4f} {m['roc_auc']:>8.4f}")
    
    # Find best model
    best_model = max(all_results.keys(), key=lambda k: all_results[k]["metrics"]["f1_score"])
    best_f1 = all_results[best_model]["metrics"]["f1_score"]
    
    print(f"\n  🏆 Best Model (by F1-Score): {best_model} ({best_f1:.4f})")
    print(f"{'='*60}")
    
    return all_results


def generate_report(all_results, output_dir=REPORTS_DIR):
    """
    Generate a comprehensive text report.
    
    Parameters
    ----------
    all_results : dict
        All model evaluation results.
    output_dir : str
        Directory to save the report.
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "evaluation_report.txt")
    
    with open(filepath, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  CREDIT CARD FRAUD DETECTION - MODEL EVALUATION REPORT\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("  Team Members:\n")
        f.write("    - Allah Yar Durrani\n")
        f.write("    - Muskan Khan\n")
        f.write("    - M. Abdullah\n")
        f.write("    - Mujtaba Khan\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("  MODEL COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        
        # Comparison table
        f.write(f"  {'Model':<25} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} "
                f"{'F1':>8} {'AUC':>8} {'MCC':>8}\n")
        f.write(f"  {'─'*78}\n")
        
        for name, results in all_results.items():
            m = results["metrics"]
            f.write(f"  {name:<25} {m['accuracy']:>9.4f} {m['precision']:>10.4f} "
                    f"{m['recall']:>8.4f} {m['f1_score']:>8.4f} {m['roc_auc']:>8.4f} "
                    f"{m['mcc']:>8.4f}\n")
        
        # Best model
        best_model = max(all_results.keys(), 
                         key=lambda k: all_results[k]["metrics"]["f1_score"])
        f.write(f"\n  🏆 Best Model: {best_model}\n\n")
        
        # Detailed results for each model
        for name, results in all_results.items():
            f.write("\n" + "─" * 70 + "\n")
            f.write(f"  {name.upper()}\n")
            f.write("─" * 70 + "\n\n")
            
            m = results["metrics"]
            f.write("  Performance Metrics:\n")
            for metric, value in m.items():
                f.write(f"    {metric:<30}: {value:.4f}\n")
            
            cm = results["confusion_matrix"]
            f.write(f"\n  Confusion Matrix:\n")
            f.write(f"    True Negatives  (TN): {cm['tn']:>8,}\n")
            f.write(f"    False Positives (FP): {cm['fp']:>8,}\n")
            f.write(f"    False Negatives (FN): {cm['fn']:>8,}\n")
            f.write(f"    True Positives  (TP): {cm['tp']:>8,}\n")
            
            biz = results["business_impact"]
            f.write(f"\n  Business Impact:\n")
            f.write(f"    Fraud caught (savings)  : ${biz['money_saved']:>12,.0f}\n")
            f.write(f"    Fraud missed (losses)   : ${biz['money_lost']:>12,.0f}\n")
            f.write(f"    False alarm costs       : ${biz['investigation_waste']:>12,.0f}\n")
            f.write(f"    Net benefit             : ${biz['net_benefit']:>12,.0f}\n")
        
        # Conclusion
        f.write("\n" + "=" * 70 + "\n")
        f.write("  CONCLUSION\n")
        f.write("=" * 70 + "\n\n")
        
        best_m = all_results[best_model]["metrics"]
        f.write(f"  The {best_model} model achieved the best F1-Score of {best_m['f1_score']:.4f}\n")
        f.write(f"  with a precision of {best_m['precision']:.4f} and recall of {best_m['recall']:.4f}.\n\n")
        f.write(f"  Key Observations:\n")
        f.write(f"  - Recall is critical: Missing fraud (False Negatives) is costlier than\n")
        f.write(f"    false alarms (False Positives).\n")
        f.write(f"  - Accuracy alone is misleading due to extreme class imbalance.\n")
        f.write(f"  - The Precision-Recall curve provides better insight than ROC for\n")
        f.write(f"    imbalanced datasets.\n")
        f.write(f"  - SMOTE significantly improved minority class detection.\n\n")
        f.write("=" * 70 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"\n  📄 Report saved to: {filepath}")
    
    # Also save JSON version for programmatic access
    json_filepath = os.path.join(output_dir, "evaluation_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for name, results in all_results.items():
        json_results[name] = {
            "metrics": results["metrics"],
            "confusion_matrix": results["confusion_matrix"],
            "business_impact": results["business_impact"],
            "training_time": results.get("training_time", 0)
        }
    
    with open(json_filepath, "w") as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"  📄 JSON results saved to: {json_filepath}")
    
    return filepath


def print_why_accuracy_fails(y_test):
    """
    Demonstrate why accuracy is a poor metric for imbalanced datasets.
    This is an educational component of the project.
    """
    print(f"\n{'='*60}")
    print(f"  📚 WHY ACCURACY FAILS FOR FRAUD DETECTION")
    print(f"{'='*60}")
    
    n_total = len(y_test)
    n_legit = (y_test == 0).sum()
    n_fraud = (y_test == 1).sum()
    
    print(f"\n  Consider our test set:")
    print(f"    Total transactions   : {n_total:,}")
    print(f"    Legitimate           : {n_legit:,} ({n_legit/n_total*100:.2f}%)")
    print(f"    Fraudulent           : {n_fraud:,} ({n_fraud/n_total*100:.2f}%)")
    
    print(f"\n  🤔 The 'Dummy Classifier' trap:")
    print(f"  If a model ALWAYS predicts 'Legitimate' (Class 0):")
    dummy_accuracy = n_legit / n_total
    print(f"    - Accuracy = {dummy_accuracy:.4f} ({dummy_accuracy*100:.2f}%)")
    print(f"    - Precision = 0.0000 (no fraud predictions, so N/A)")
    print(f"    - Recall = 0.0000 (catches 0 out of {n_fraud:,} frauds)")
    print(f"    - F1-Score = 0.0000")
    print(f"\n  📊 This is why we prioritize:")
    print(f"    1. Recall (Sensitivity)  - How many frauds did we catch?")
    print(f"    2. Precision             - Of flagged transactions, how many were fraud?")
    print(f"    3. F1-Score              - Harmonic mean of precision and recall")
    print(f"    4. ROC-AUC              - Overall discriminative ability")
    print(f"    5. Precision-Recall AUC  - Best for imbalanced datasets")
    print(f"\n  ⚠️  In banking: Missing 1 fraud = potential $1000s loss")
    print(f"     False alarm = just $10 investigation cost")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("Evaluation module loaded. Use evaluate_all_models() to evaluate.")
