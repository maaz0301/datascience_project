"""
Data Loader Module
==================
Handles loading the credit card fraud dataset and performing
initial exploratory data analysis (EDA).
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_PATH, TARGET_COLUMN


def load_dataset(filepath=DATASET_PATH):
    """
    Load the credit card fraud dataset from CSV.
    
    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    
    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    
    Raises
    ------
    FileNotFoundError
        If the dataset file doesn't exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"\n❌ Dataset not found at: {filepath}\n"
            f"   Please run: python generate_dataset.py\n"
            f"   Or download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
        )
    
    print(f"\n📂 Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"   ✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df


def explore_dataset(df):
    """
    Perform comprehensive exploratory data analysis.
    
    Parameters
    ----------
    df : pd.DataFrame
        The credit card fraud dataset.
    
    Returns
    -------
    dict
        Dictionary containing all EDA results.
    """
    print(f"\n{'='*70}")
    print(f"  📊 EXPLORATORY DATA ANALYSIS (EDA)")
    print(f"{'='*70}")
    
    eda_results = {}
    
    # ---- 1. Basic Information ----
    print(f"\n  1️⃣  BASIC INFORMATION")
    print(f"  {'─'*50}")
    print(f"  Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory usage   : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"  Data types     :")
    for dtype, count in df.dtypes.value_counts().items():
        print(f"    {dtype}: {count} columns")
    
    eda_results["shape"] = df.shape
    eda_results["memory_mb"] = df.memory_usage(deep=True).sum() / 1024**2
    
    # ---- 2. Missing Values ----
    print(f"\n  2️⃣  MISSING VALUES")
    print(f"  {'─'*50}")
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print(f"  ✅ No missing values found!")
    else:
        print(f"  ⚠️  Total missing values: {total_missing}")
        for col in missing[missing > 0].index:
            print(f"    {col}: {missing[col]:,} ({missing[col]/len(df)*100:.2f}%)")
    
    eda_results["missing_values"] = missing
    
    # ---- 3. Duplicate Rows ----
    print(f"\n  3️⃣  DUPLICATE ROWS")
    print(f"  {'─'*50}")
    duplicates = df.duplicated().sum()
    print(f"  Duplicate rows: {duplicates:,} ({duplicates/len(df)*100:.4f}%)")
    eda_results["duplicates"] = duplicates
    
    # ---- 4. Class Distribution ----
    print(f"\n  4️⃣  CLASS DISTRIBUTION (Target Variable)")
    print(f"  {'─'*50}")
    class_dist = df[TARGET_COLUMN].value_counts().sort_index()
    class_pct = df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100
    
    labels = {0: "Legitimate", 1: "Fraud"}
    for cls in class_dist.index:
        print(f"  Class {cls} ({labels.get(cls, 'Unknown')}): "
              f"{class_dist[cls]:>8,} transactions ({class_pct[cls]:.3f}%)")
    
    imbalance_ratio = class_dist[0] / class_dist[1]
    print(f"\n  ⚖️  Imbalance Ratio: {imbalance_ratio:.0f}:1 (Legitimate:Fraud)")
    
    eda_results["class_distribution"] = class_dist
    eda_results["class_percentages"] = class_pct
    eda_results["imbalance_ratio"] = imbalance_ratio
    
    # ---- 5. Statistical Summary ----
    print(f"\n  5️⃣  STATISTICAL SUMMARY")
    print(f"  {'─'*50}")
    
    # Key features summary
    key_features = ["Time", "Amount"]
    for feat in key_features:
        if feat in df.columns:
            print(f"\n  {feat}:")
            print(f"    Mean   : {df[feat].mean():>12.2f}")
            print(f"    Std    : {df[feat].std():>12.2f}")
            print(f"    Min    : {df[feat].min():>12.2f}")
            print(f"    25%    : {df[feat].quantile(0.25):>12.2f}")
            print(f"    Median : {df[feat].median():>12.2f}")
            print(f"    75%    : {df[feat].quantile(0.75):>12.2f}")
            print(f"    Max    : {df[feat].max():>12.2f}")
    
    eda_results["statistics"] = df.describe()
    
    # ---- 6. Feature Correlations with Target ----
    print(f"\n  6️⃣  TOP FEATURES CORRELATED WITH FRAUD")
    print(f"  {'─'*50}")
    
    correlations = df.corr()[TARGET_COLUMN].drop(TARGET_COLUMN).abs().sort_values(ascending=False)
    top_corr = correlations.head(10)
    
    for feat, corr in top_corr.items():
        direction = "+" if df.corr()[TARGET_COLUMN][feat] > 0 else "-"
        bar = "█" * int(corr * 40)
        print(f"  {feat:>8}: {direction}{corr:.4f} {bar}")
    
    eda_results["correlations"] = correlations
    
    # ---- 7. Amount Analysis by Class ----
    print(f"\n  7️⃣  AMOUNT ANALYSIS BY CLASS")
    print(f"  {'─'*50}")
    
    for cls in [0, 1]:
        subset = df[df[TARGET_COLUMN] == cls]["Amount"]
        print(f"\n  {labels[cls]} transactions:")
        print(f"    Count  : {len(subset):>10,}")
        print(f"    Mean   : ${subset.mean():>10.2f}")
        print(f"    Median : ${subset.median():>10.2f}")
        print(f"    Max    : ${subset.max():>10.2f}")
    
    eda_results["amount_by_class"] = {
        cls: df[df[TARGET_COLUMN] == cls]["Amount"].describe()
        for cls in [0, 1]
    }
    
    print(f"\n{'='*70}")
    print(f"  ✅ EDA Complete!")
    print(f"{'='*70}")
    
    return eda_results


if __name__ == "__main__":
    # Standalone test
    df = load_dataset()
    results = explore_dataset(df)
