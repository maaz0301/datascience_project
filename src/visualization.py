"""
Visualization Module
====================
Production-ready visualizations for the Credit Card Fraud Detection project.
All plots are saved to the reports/figures/ directory.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURES_DIR, FIGURE_DPI, COLOR_PALETTE, PLOT_COLORS, TARGET_COLUMN

# Set global style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.titlesize': 16,
    'figure.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#dee2e6',
    'grid.color': '#e9ecef',
    'grid.alpha': 0.7,
})


def save_figure(fig, filename):
    """Save figure to the figures directory."""
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"    📊 Saved: {filepath}")
    return filepath


# ============================================================================
# 1. EXPLORATORY DATA ANALYSIS PLOTS
# ============================================================================

def plot_class_distribution(df, target_col=TARGET_COLUMN):
    """Plot the class distribution (imbalanced dataset visualization)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Class Distribution Analysis", fontsize=16, fontweight='bold', y=1.02)
    
    class_counts = df[target_col].value_counts().sort_index()
    labels = ["Legitimate\n(Class 0)", "Fraud\n(Class 1)"]
    colors = [COLOR_PALETTE["legitimate"], COLOR_PALETTE["fraud"]]
    
    # Bar chart
    bars = axes[0].bar(labels, class_counts.values, color=colors, 
                       edgecolor='white', linewidth=2, width=0.6)
    for bar, count in zip(bars, class_counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 500,
                     f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    axes[0].set_title("Transaction Count by Class")
    axes[0].set_ylabel("Number of Transactions")
    axes[0].grid(axis='y', alpha=0.3)
    
    # Pie chart
    axes[1].pie(class_counts.values, labels=labels, colors=colors,
                autopct=lambda pct: f'{pct:.2f}%\n({int(pct/100*sum(class_counts.values)):,})',
                shadow=True, startangle=90, explode=(0, 0.1),
                textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title("Class Proportion")
    
    # Log scale bar chart (to see both classes clearly)
    bars = axes[2].bar(labels, class_counts.values, color=colors,
                       edgecolor='white', linewidth=2, width=0.6)
    axes[2].set_yscale('log')
    for bar, count in zip(bars, class_counts.values):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
                     f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    axes[2].set_title("Transaction Count (Log Scale)")
    axes[2].set_ylabel("Number of Transactions (log)")
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return save_figure(fig, "01_class_distribution.png")


def plot_amount_distribution(df, target_col=TARGET_COLUMN):
    """Plot transaction amount distributions for legitimate vs fraud."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Transaction Amount Analysis", fontsize=16, fontweight='bold', y=1.02)
    
    legit = df[df[target_col] == 0]["Amount"]
    fraud = df[df[target_col] == 1]["Amount"]
    
    # Distribution of all amounts
    axes[0, 0].hist(df["Amount"], bins=100, color=COLOR_PALETTE["primary"], 
                     alpha=0.7, edgecolor='white')
    axes[0, 0].set_title("Overall Amount Distribution")
    axes[0, 0].set_xlabel("Amount ($)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].axvline(df["Amount"].mean(), color='red', linestyle='--', 
                        label=f'Mean: ${df["Amount"].mean():.2f}')
    axes[0, 0].legend()
    
    # Legitimate vs Fraud comparison
    axes[0, 1].hist(legit, bins=80, alpha=0.6, color=COLOR_PALETTE["legitimate"], 
                     label=f'Legitimate (n={len(legit):,})', edgecolor='white')
    axes[0, 1].hist(fraud, bins=80, alpha=0.8, color=COLOR_PALETTE["fraud"], 
                     label=f'Fraud (n={len(fraud):,})', edgecolor='white')
    axes[0, 1].set_title("Amount: Legitimate vs Fraud")
    axes[0, 1].set_xlabel("Amount ($)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()
    
    # Box plots by class
    box_data = [legit.values, fraud.values]
    bp = axes[1, 0].boxplot(box_data, labels=["Legitimate", "Fraud"],
                             patch_artist=True, showfliers=True)
    bp['boxes'][0].set_facecolor(COLOR_PALETTE["legitimate"])
    bp['boxes'][1].set_facecolor(COLOR_PALETTE["fraud"])
    for box in bp['boxes']:
        box.set_alpha(0.7)
    axes[1, 0].set_title("Amount Distribution by Class")
    axes[1, 0].set_ylabel("Amount ($)")
    
    # Log-transformed amount distribution
    axes[1, 1].hist(np.log1p(legit), bins=60, alpha=0.6, 
                     color=COLOR_PALETTE["legitimate"], label='Legitimate', edgecolor='white')
    axes[1, 1].hist(np.log1p(fraud), bins=60, alpha=0.8, 
                     color=COLOR_PALETTE["fraud"], label='Fraud', edgecolor='white')
    axes[1, 1].set_title("Log-Transformed Amount Distribution")
    axes[1, 1].set_xlabel("Log(Amount + 1)")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].legend()
    
    plt.tight_layout()
    return save_figure(fig, "02_amount_distribution.png")


def plot_time_distribution(df, target_col=TARGET_COLUMN):
    """Plot transaction time distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Transaction Time Analysis", fontsize=16, fontweight='bold', y=1.02)
    
    legit = df[df[target_col] == 0]
    fraud = df[df[target_col] == 1]
    
    # Time distribution
    axes[0].hist(legit["Time"] / 3600, bins=48, alpha=0.6, 
                  color=COLOR_PALETTE["legitimate"], label='Legitimate', edgecolor='white')
    axes[0].hist(fraud["Time"] / 3600, bins=48, alpha=0.8, 
                  color=COLOR_PALETTE["fraud"], label='Fraud', edgecolor='white')
    axes[0].set_title("Transaction Distribution Over Time")
    axes[0].set_xlabel("Time (Hours)")
    axes[0].set_ylabel("Number of Transactions")
    axes[0].legend()
    
    # Fraud rate over time
    df_temp = df.copy()
    df_temp["Hour_Bin"] = (df_temp["Time"] / 3600).astype(int)
    hourly_fraud = df_temp.groupby("Hour_Bin")[target_col].mean() * 100
    
    axes[1].plot(hourly_fraud.index, hourly_fraud.values, 
                  color=COLOR_PALETTE["fraud"], linewidth=2, marker='o', markersize=4)
    axes[1].fill_between(hourly_fraud.index, hourly_fraud.values, 
                          alpha=0.2, color=COLOR_PALETTE["fraud"])
    axes[1].set_title("Fraud Rate Over Time")
    axes[1].set_xlabel("Time (Hours)")
    axes[1].set_ylabel("Fraud Rate (%)")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return save_figure(fig, "03_time_distribution.png")


def plot_correlation_heatmap(df, target_col=TARGET_COLUMN):
    """Plot correlation heatmap for top features."""
    # Select top 15 features most correlated with target
    correlations = df.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
    top_features = correlations.head(15).index.tolist()
    
    corr_matrix = df[top_features + [target_col]].corr()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                annot=True, fmt='.2f', square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={"size": 8})
    
    ax.set_title("Correlation Heatmap (Top 15 Features + Target)", 
                  fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return save_figure(fig, "04_correlation_heatmap.png")


def plot_feature_distributions(df, target_col=TARGET_COLUMN):
    """Plot distributions of key V-features for legitimate vs fraud."""
    features = ["V1", "V3", "V4", "V7", "V10", "V11", "V12", "V14", "V16", "V17"]
    features = [f for f in features if f in df.columns]
    
    n_features = len(features)
    n_cols = 5
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    fig.suptitle("Feature Distributions: Legitimate vs Fraud", 
                  fontsize=16, fontweight='bold', y=1.02)
    
    axes_flat = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes_flat[idx]
        
        legit_data = df[df[target_col] == 0][feature]
        fraud_data = df[df[target_col] == 1][feature]
        
        ax.hist(legit_data, bins=50, alpha=0.5, color=COLOR_PALETTE["legitimate"],
                label='Legitimate', density=True, edgecolor='white')
        ax.hist(fraud_data, bins=50, alpha=0.7, color=COLOR_PALETTE["fraud"],
                label='Fraud', density=True, edgecolor='white')
        
        ax.set_title(feature, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(features), len(axes_flat)):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    return save_figure(fig, "05_feature_distributions.png")


# ============================================================================
# 2. MODEL EVALUATION PLOTS
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
    """Plot a detailed confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix - {model_name}", fontsize=14, fontweight='bold')
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                annot_kws={"size": 14, "fontweight": "bold"})
    axes[0].set_title("Raw Counts")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")
    
    # Normalized (percentages)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Reds', ax=axes[1],
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                annot_kws={"size": 14, "fontweight": "bold"})
    axes[1].set_title("Normalized (Row %)")
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")
    
    # Add annotation
    tn, fp, fn, tp = cm.ravel()
    fig.text(0.5, -0.05, 
             f"TN={tn:,}  |  FP={fp:,}  |  FN={fn:,}  |  TP={tp:,}  |  "
             f"Recall={tp/(tp+fn):.4f}  |  Precision={tp/(tp+fp):.4f}",
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return save_figure(fig, f"06_confusion_matrix_{model_name.lower().replace(' ', '_')}.png")


def plot_roc_curves(results_dict, y_test):
    """
    Plot ROC curves for multiple models.
    
    Parameters
    ----------
    results_dict : dict
        {model_name: {"y_pred_proba": probabilities, ...}}
    y_test : array-like
        True labels.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for idx, (name, results) in enumerate(results_dict.items()):
        y_proba = results.get("y_pred_proba")
        if y_proba is None:
            continue
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=PLOT_COLORS[idx % len(PLOT_COLORS)],
                linewidth=2.5, label=f'{name} (AUC = {roc_auc:.4f})')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5,
            label='Random Classifier (AUC = 0.5000)')
    
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curves - Model Comparison", fontsize=15, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Shade the area for the best model
    best_model = max(results_dict.keys(), 
                     key=lambda k: auc(*roc_curve(y_test, results_dict[k]["y_pred_proba"])[:2])
                     if results_dict[k].get("y_pred_proba") is not None else 0)
    best_proba = results_dict[best_model]["y_pred_proba"]
    if best_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, best_proba)
        ax.fill_between(fpr, tpr, alpha=0.1, color=PLOT_COLORS[0])
    
    plt.tight_layout()
    return save_figure(fig, "07_roc_curves.png")


def plot_precision_recall_curves(results_dict, y_test):
    """
    Plot Precision-Recall curves for multiple models.
    This is especially important for imbalanced datasets.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for idx, (name, results) in enumerate(results_dict.items()):
        y_proba = results.get("y_pred_proba")
        if y_proba is None:
            continue
        
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        
        ax.plot(recall, precision, color=PLOT_COLORS[idx % len(PLOT_COLORS)],
                linewidth=2.5, label=f'{name} (AP = {ap:.4f})')
    
    # Baseline (proportion of positives)
    baseline = y_test.sum() / len(y_test)
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5,
               label=f'Baseline (Prevalence = {baseline:.4f})')
    
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curves - Model Comparison\n"
                 "(More Important Than ROC for Imbalanced Data)", 
                 fontsize=15, fontweight='bold')
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return save_figure(fig, "08_precision_recall_curves.png")


def plot_model_comparison(results_dict):
    """Plot a comprehensive model comparison dashboard."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    model_names = list(results_dict.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Model Performance Comparison", fontsize=16, fontweight='bold', y=1.02)
    
    # Grouped bar chart
    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    
    for idx, name in enumerate(model_names):
        metrics_values = results_dict[name].get("metrics", {})
        values = [
            metrics_values.get("accuracy", 0),
            metrics_values.get("precision", 0),
            metrics_values.get("recall", 0),
            metrics_values.get("f1_score", 0),
            metrics_values.get("roc_auc", 0),
        ]
        
        offset = (idx - len(model_names)/2 + 0.5) * width
        bars = axes[0].bar(x + offset, values, width * 0.9,
                           label=name, color=PLOT_COLORS[idx % len(PLOT_COLORS)],
                           edgecolor='white', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val > 0:
                axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=8,
                            fontweight='bold', rotation=45)
    
    axes[0].set_xlabel("Metrics")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Performance Metrics by Model")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics)
    axes[0].legend(loc='lower left')
    axes[0].set_ylim(0, 1.15)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Radar chart
    ax_radar = fig.add_subplot(122, projection='polar')
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, name in enumerate(model_names):
        metrics_values = results_dict[name].get("metrics", {})
        values = [
            metrics_values.get("accuracy", 0),
            metrics_values.get("precision", 0),
            metrics_values.get("recall", 0),
            metrics_values.get("f1_score", 0),
            metrics_values.get("roc_auc", 0),
        ]
        values += values[:1]
        
        ax_radar.plot(angles, values, 'o-', linewidth=2,
                      color=PLOT_COLORS[idx % len(PLOT_COLORS)], label=name)
        ax_radar.fill(angles, values, alpha=0.1,
                      color=PLOT_COLORS[idx % len(PLOT_COLORS)])
    
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(metrics, fontsize=10)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title("Model Comparison Radar", fontsize=14, fontweight='bold', pad=20)
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Remove the original axes[1] since we used polar
    axes[1].set_visible(False)
    
    plt.tight_layout()
    return save_figure(fig, "09_model_comparison.png")


def plot_feature_importance(importances, feature_names, model_name="Model", top_n=20):
    """Plot feature importances from tree-based models."""
    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
    
    bars = ax.barh(range(len(top_features)), top_importances,
                   color=colors, edgecolor='white', linewidth=1)
    
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances - {model_name}", 
                  fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, imp in zip(bars, top_importances):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2.,
                f'{imp:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    return save_figure(fig, f"10_feature_importance_{model_name.lower().replace(' ', '_')}.png")


def plot_threshold_analysis(y_true, y_proba, model_name="Model"):
    """Analyze the impact of different classification thresholds."""
    thresholds = np.arange(0.1, 0.95, 0.05)
    
    precisions, recalls, f1_scores = [], [], []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(thresholds, precisions, 'b-o', linewidth=2, markersize=5, label='Precision')
    ax.plot(thresholds, recalls, 'r-s', linewidth=2, markersize=5, label='Recall')
    ax.plot(thresholds, f1_scores, 'g-^', linewidth=2, markersize=5, label='F1-Score')
    
    # Mark optimal F1 threshold
    optimal_idx = np.argmax(f1_scores)
    optimal_thresh = thresholds[optimal_idx]
    ax.axvline(x=optimal_thresh, color='gray', linestyle='--', alpha=0.7,
               label=f'Optimal Threshold = {optimal_thresh:.2f}')
    ax.scatter([optimal_thresh], [f1_scores[optimal_idx]], s=200, c='gold',
              edgecolors='black', zorder=5, marker='*')
    
    ax.set_xlabel("Classification Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Threshold Analysis - {model_name}\n"
                 f"(Optimal F1 at threshold = {optimal_thresh:.2f})",
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.05, 0.95])
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    return save_figure(fig, f"11_threshold_analysis_{model_name.lower().replace(' ', '_')}.png")


def generate_all_eda_plots(df):
    """Generate all EDA visualizations."""
    print(f"\n{'='*60}")
    print(f"  📊 GENERATING EDA VISUALIZATIONS")
    print(f"{'='*60}")
    
    plots = []
    
    print(f"\n  1/5 Class Distribution...")
    plots.append(plot_class_distribution(df))
    
    print(f"\n  2/5 Amount Distribution...")
    plots.append(plot_amount_distribution(df))
    
    print(f"\n  3/5 Time Distribution...")
    plots.append(plot_time_distribution(df))
    
    print(f"\n  4/5 Correlation Heatmap...")
    plots.append(plot_correlation_heatmap(df))
    
    print(f"\n  5/5 Feature Distributions...")
    plots.append(plot_feature_distributions(df))
    
    print(f"\n  ✅ All EDA plots generated! ({len(plots)} figures)")
    print(f"  📁 Saved to: {FIGURES_DIR}")
    
    return plots


if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    generate_all_eda_plots(df)
