"""
Data Preprocessing Module
=========================
Handles data cleaning, scaling, and train-test splitting.
Prepares the dataset for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    TARGET_COLUMN, RANDOM_STATE, TEST_SIZE,
    SCALING_COLUMNS, PCA_FEATURES
)


def check_data_quality(df):
    """
    Check data quality and report issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    
    Returns
    -------
    dict
        Quality report with identified issues.
    """
    print(f"\n{'='*60}")
    print(f"  🔍 DATA QUALITY CHECK")
    print(f"{'='*60}")
    
    quality_report = {
        "missing_values": {},
        "infinite_values": {},
        "outliers": {},
        "issues_found": False
    }
    
    # Check missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        quality_report["missing_values"] = missing[missing > 0].to_dict()
        quality_report["issues_found"] = True
        print(f"  ⚠️  Missing values detected:")
        for col, count in quality_report["missing_values"].items():
            print(f"      {col}: {count}")
    else:
        print(f"  ✅ No missing values")
    
    # Check infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            quality_report["infinite_values"][col] = inf_count
            quality_report["issues_found"] = True
    
    if quality_report["infinite_values"]:
        print(f"  ⚠️  Infinite values detected:")
        for col, count in quality_report["infinite_values"].items():
            print(f"      {col}: {count}")
    else:
        print(f"  ✅ No infinite values")
    
    # Check for extreme outliers in Amount
    if "Amount" in df.columns:
        Q1 = df["Amount"].quantile(0.25)
        Q3 = df["Amount"].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df["Amount"] < Q1 - 3 * IQR) | (df["Amount"] > Q3 + 3 * IQR)).sum()
        quality_report["outliers"]["Amount"] = outlier_count
        print(f"  📊 Amount outliers (3×IQR): {outlier_count:,}")
    
    # Check class balance info
    class_counts = df[TARGET_COLUMN].value_counts()
    minority_pct = class_counts.min() / len(df) * 100
    print(f"  ⚖️  Minority class: {minority_pct:.3f}% of total")
    
    if not quality_report["issues_found"]:
        print(f"\n  ✅ No critical data quality issues found!")
    
    print(f"{'='*60}")
    
    return quality_report


def handle_missing_values(df, strategy="median"):
    """
    Handle missing values in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    strategy : str
        Strategy for handling missing values: 'median', 'mean', or 'drop'.
    
    Returns
    -------
    pd.DataFrame
        Dataset with missing values handled.
    """
    if df.isnull().sum().sum() == 0:
        print("  ✅ No missing values to handle.")
        return df
    
    df_clean = df.copy()
    
    if strategy == "drop":
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        dropped = initial_rows - len(df_clean)
        print(f"  🗑️  Dropped {dropped:,} rows with missing values")
    elif strategy in ["median", "mean"]:
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                fill_value = df_clean[col].median() if strategy == "median" else df_clean[col].mean()
                df_clean[col].fillna(fill_value, inplace=True)
                print(f"  📝 Filled {col} missing values with {strategy}: {fill_value:.4f}")
    
    return df_clean


def remove_duplicates(df):
    """
    Remove duplicate rows from the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    
    Returns
    -------
    pd.DataFrame
        Dataset with duplicates removed.
    """
    initial_rows = len(df)
    df_clean = df.drop_duplicates()
    removed = initial_rows - len(df_clean)
    
    if removed > 0:
        print(f"  🗑️  Removed {removed:,} duplicate rows")
    else:
        print(f"  ✅ No duplicate rows found")
    
    return df_clean


def scale_features(df, columns=None, method="robust"):
    """
    Scale numerical features using specified method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list, optional
        Columns to scale. Defaults to SCALING_COLUMNS.
    method : str
        Scaling method: 'standard' (StandardScaler) or 'robust' (RobustScaler).
        RobustScaler is preferred for data with outliers.
    
    Returns
    -------
    tuple
        (pd.DataFrame, dict of scalers)
    """
    if columns is None:
        columns = SCALING_COLUMNS
    
    df_scaled = df.copy()
    scalers = {}
    
    print(f"\n  📏 Scaling features using {method} method:")
    
    for col in columns:
        if col in df_scaled.columns:
            if method == "robust":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            
            df_scaled[col] = scaler.fit_transform(df_scaled[[col]])
            scalers[col] = scaler
            print(f"    ✅ {col}: mean={df_scaled[col].mean():.4f}, std={df_scaled[col].std():.4f}")
    
    return df_scaled, scalers


def split_data(df, target_column=TARGET_COLUMN, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split data into training and testing sets with stratification.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset.
    target_column : str
        Name of the target column.
    test_size : float
        Proportion of data for testing.
    random_state : int
        Random seed.
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Maintains class distribution in both sets
    )
    
    print(f"\n  📊 DATA SPLIT (Stratified)")
    print(f"  {'─'*45}")
    print(f"  Training set : {X_train.shape[0]:>8,} samples ({(1-test_size)*100:.0f}%)")
    print(f"    - Legitimate: {(y_train == 0).sum():>8,}")
    print(f"    - Fraud     : {(y_train == 1).sum():>8,}")
    print(f"  Testing set  : {X_test.shape[0]:>8,} samples ({test_size*100:.0f}%)")
    print(f"    - Legitimate: {(y_test == 0).sum():>8,}")
    print(f"    - Fraud     : {(y_test == 1).sum():>8,}")
    print(f"  Features     : {X_train.shape[1]:>8}")
    
    return X_train, X_test, y_train, y_test


def preprocess_pipeline(df):
    """
    Run the complete preprocessing pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, scalers, preprocessed_df)
    """
    print(f"\n{'='*60}")
    print(f"  🔧 PREPROCESSING PIPELINE")
    print(f"{'='*60}")
    
    # Step 1: Data Quality Check
    quality_report = check_data_quality(df)
    
    # Step 2: Handle Missing Values
    print(f"\n  Step 1: Handle Missing Values")
    df_clean = handle_missing_values(df, strategy="median")
    
    # Step 3: Remove Duplicates
    print(f"\n  Step 2: Remove Duplicates")
    df_clean = remove_duplicates(df_clean)
    
    # Step 4: Scale Features
    print(f"\n  Step 3: Feature Scaling")
    df_scaled, scalers = scale_features(df_clean, method="robust")
    
    # Step 5: Split Data
    print(f"\n  Step 4: Train-Test Split")
    X_train, X_test, y_train, y_test = split_data(df_scaled)
    
    print(f"\n{'='*60}")
    print(f"  ✅ Preprocessing Complete!")
    print(f"{'='*60}")
    
    return X_train, X_test, y_train, y_test, scalers, df_scaled


if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    X_train, X_test, y_train, y_test, scalers, df_scaled = preprocess_pipeline(df)
