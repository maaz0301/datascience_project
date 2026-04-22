"""
Feature Engineering Module
==========================
Creates new features from existing data to improve model performance.
Focuses on time-based features and transaction pattern analysis.
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TARGET_COLUMN


def create_time_features(df, verbose=True):
    """
    Create time-based features from the 'Time' column.
    
    The 'Time' column represents seconds elapsed since the first transaction.
    We extract cyclical hour-of-day features to capture daily patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'Time' column.
    verbose : bool
        Whether to print success messages.
    
    Returns
    -------
    pd.DataFrame
        Dataset with new time features added.
    """
    df_feat = df.copy()
    
    if "Time" not in df_feat.columns:
        if verbose: print("  ⚠️  'Time' column not found. Skipping time features.")
        return df_feat
    
    # Convert seconds to hours (cyclical within 24-hour period)
    df_feat["Hour"] = (df_feat["Time"] / 3600) % 24
    
    # Cyclical encoding (sine and cosine) to capture the circular nature of time
    df_feat["Hour_sin"] = np.sin(2 * np.pi * df_feat["Hour"] / 24)
    df_feat["Hour_cos"] = np.cos(2 * np.pi * df_feat["Hour"] / 24)
    
    # Time period categories
    def get_time_period(hour):
        if 6 <= hour < 12:
            return 0  # Morning
        elif 12 <= hour < 18:
            return 1  # Afternoon
        elif 18 <= hour < 22:
            return 2  # Evening
        else:
            return 3  # Night (high risk period)
    
    df_feat["Time_Period"] = df_feat["Hour"].apply(get_time_period)
    
    # Is nighttime (potentially higher fraud risk)
    df_feat["Is_Night"] = ((df_feat["Hour"] >= 22) | (df_feat["Hour"] < 6)).astype(int)
    
    if verbose:
        print(f"    ✅ Created time features: Hour, Hour_sin, Hour_cos, Time_Period, Is_Night")
    
    return df_feat


def create_amount_features(df, verbose=True):
    """
    Create amount-based features to capture transaction patterns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with 'Amount' column.
    verbose : bool
        Whether to print success messages.
    
    Returns
    -------
    pd.DataFrame
        Dataset with new amount features added.
    """
    df_feat = df.copy()
    
    if "Amount" not in df_feat.columns:
        if verbose: print("  ⚠️  'Amount' column not found. Skipping amount features.")
        return df_feat
    
    # Log transform of amount (reduces skewness)
    df_feat["Amount_Log"] = np.log1p(df_feat["Amount"].abs())
    
    # Amount bins (categorical)
    amount_bins = [0, 10, 50, 100, 500, 1000, float('inf')]
    amount_labels = [0, 1, 2, 3, 4, 5]
    df_feat["Amount_Bin"] = pd.cut(
        df_feat["Amount"],
        bins=amount_bins,
        labels=amount_labels,
        include_lowest=True
    ).fillna(0).astype(int)
    
    # Is the amount a round number (potential fraud indicator)
    df_feat["Is_Round_Amount"] = (df_feat["Amount"] % 10 == 0).astype(int)
    
    # Amount relative to overall statistics
    amount_mean = df_feat["Amount"].mean()
    amount_std = df_feat["Amount"].std()
    df_feat["Amount_ZScore"] = (df_feat["Amount"] - amount_mean) / (amount_std + 1e-8)
    
    # Is high value transaction (above 95th percentile)
    threshold_95 = df_feat["Amount"].quantile(0.95)
    df_feat["Is_High_Amount"] = (df_feat["Amount"] > threshold_95).astype(int)
    
    if verbose:
        print(f"    ✅ Created amount features: Amount_Log, Amount_Bin, Is_Round_Amount, "
              f"Amount_ZScore, Is_High_Amount")
    
    return df_feat


def create_pca_interaction_features(df, verbose=True):
    """
    Create interaction features from PCA components.
    High-importance PCA features can be combined to create new signals.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with V1-V28 columns.
    verbose : bool
        Whether to print success messages.
    
    Returns
    -------
    pd.DataFrame
        Dataset with interaction features added.
    """
    df_feat = df.copy()
    
    # Top correlated features with fraud (from typical analysis)
    important_features = ["V1", "V3", "V4", "V7", "V10", "V11", "V12", "V14", "V16", "V17"]
    
    available_features = [f for f in important_features if f in df_feat.columns]
    
    if len(available_features) < 2:
        if verbose: print("  ⚠️  Not enough PCA features for interactions.")
        return df_feat
    
    # Magnitude of important PCA features (L2 norm)
    df_feat["PCA_Magnitude"] = np.sqrt(
        sum(df_feat[f] ** 2 for f in available_features)
    )
    
    # Mean of important features
    df_feat["PCA_Mean"] = df_feat[available_features].mean(axis=1)
    
    # Standard deviation of important features (variability indicator)
    df_feat["PCA_Std"] = df_feat[available_features].std(axis=1)
    
    # Key interactions (most discriminative pairs)
    if "V14" in df_feat.columns and "V4" in df_feat.columns:
        df_feat["V14_x_V4"] = df_feat["V14"] * df_feat["V4"]
    
    if "V12" in df_feat.columns and "V10" in df_feat.columns:
        df_feat["V12_x_V10"] = df_feat["V12"] * df_feat["V10"]
    
    if "V17" in df_feat.columns and "V14" in df_feat.columns:
        df_feat["V17_x_V14"] = df_feat["V17"] * df_feat["V14"]
    
    if verbose:
        print(f"    ✅ Created PCA interaction features: PCA_Magnitude, PCA_Mean, PCA_Std, "
              f"and key interactions")
    
    return df_feat


def select_features(df, target_column=TARGET_COLUMN, correlation_threshold=0.01):
    """
    Select features based on correlation with target.
    Removes features with very low correlation to the target variable.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset with all features.
    target_column : str
        Name of target column.
    correlation_threshold : float
        Minimum absolute correlation to keep a feature.
    
    Returns
    -------
    tuple
        (selected_df, selected_features, dropped_features)
    """
    correlations = df.corr()[target_column].drop(target_column).abs()
    
    selected = correlations[correlations >= correlation_threshold].index.tolist()
    dropped = correlations[correlations < correlation_threshold].index.tolist()
    
    print(f"\n  📋 FEATURE SELECTION")
    print(f"  {'─'*45}")
    print(f"  Total features     : {len(correlations)}")
    print(f"  Selected features  : {len(selected)}")
    print(f"  Dropped features   : {len(dropped)}")
    
    if dropped:
        print(f"  Dropped (low corr) : {', '.join(dropped[:5])}")
        if len(dropped) > 5:
            print(f"                       ... and {len(dropped) - 5} more")
    
    selected_df = df[selected + [target_column]]
    
    return selected_df, selected, dropped


def feature_engineering_pipeline(df):
    """
    Run the complete feature engineering pipeline.
    
    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed dataset.
    
    Returns
    -------
    pd.DataFrame
        Dataset with engineered features.
    """
    print(f"\n{'='*60}")
    print(f"  ⚙️  FEATURE ENGINEERING PIPELINE")
    print(f"{'='*60}")
    
    initial_features = df.shape[1]
    
    # Step 1: Time features
    print(f"\n  Step 1: Time Features")
    df = create_time_features(df)
    
    # Step 2: Amount features
    print(f"\n  Step 2: Amount Features")
    df = create_amount_features(df)
    
    # Step 3: PCA interaction features
    print(f"\n  Step 3: PCA Interaction Features")
    df = create_pca_interaction_features(df)
    
    final_features = df.shape[1]
    new_features = final_features - initial_features
    
    print(f"\n  📊 Feature Engineering Summary:")
    print(f"  {'─'*45}")
    print(f"  Original features  : {initial_features}")
    print(f"  New features       : {new_features}")
    print(f"  Total features     : {final_features}")
    
    print(f"\n{'='*60}")
    print(f"  ✅ Feature Engineering Complete!")
    print(f"{'='*60}")
    
    return df


if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    df_engineered = feature_engineering_pipeline(df)
    print(f"\nNew columns: {list(df_engineered.columns)}")
