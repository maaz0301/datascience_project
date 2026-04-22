"""
Synthetic Credit Card Fraud Dataset Generator
==============================================
Generates a synthetic dataset that mimics the structure and statistical
properties of the real Kaggle Credit Card Fraud Detection dataset.

The real dataset contains:
  - Time: Seconds elapsed between each transaction and the first transaction
  - V1-V28: PCA-transformed features (anonymized)
  - Amount: Transaction amount
  - Class: 0 = Legitimate, 1 = Fraud

Usage:
    python generate_dataset.py

Note:
    For the REAL dataset, download from:
    https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    Place the 'creditcard.csv' file in the 'data/' directory.
"""

import numpy as np
import pandas as pd
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, DATASET_PATH, SYNTHETIC_NUM_TRANSACTIONS, SYNTHETIC_FRAUD_RATIO, RANDOM_STATE


def generate_synthetic_dataset(
    n_transactions=SYNTHETIC_NUM_TRANSACTIONS,
    fraud_ratio=SYNTHETIC_FRAUD_RATIO,
    random_state=RANDOM_STATE
):
    """
    Generate a synthetic credit card fraud dataset.
    
    Parameters
    ----------
    n_transactions : int
        Total number of transactions to generate.
    fraud_ratio : float
        Proportion of fraudulent transactions (0 to 1).
    random_state : int
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        Synthetic dataset with columns: Time, V1-V28, Amount, Class
    """
    np.random.seed(random_state)
    
    n_fraud = int(n_transactions * fraud_ratio)
    n_legit = n_transactions - n_fraud
    
    print(f"{'='*60}")
    print(f"  SYNTHETIC DATASET GENERATOR")
    print(f"{'='*60}")
    print(f"  Total transactions : {n_transactions:,}")
    print(f"  Legitimate         : {n_legit:,} ({(1-fraud_ratio)*100:.2f}%)")
    print(f"  Fraudulent         : {n_fraud:,} ({fraud_ratio*100:.2f}%)")
    print(f"{'='*60}")
    
    # ---- Generate Time feature ----
    # Time spans ~48 hours (172,800 seconds) as in the real dataset
    time_legit = np.sort(np.random.uniform(0, 172800, n_legit))
    time_fraud = np.random.uniform(0, 172800, n_fraud)
    
    # ---- Generate PCA features V1-V28 ----
    # Legitimate transactions: centered around 0 with unit variance
    v_legit = np.random.randn(n_legit, 28)
    
    # Fraudulent transactions: shifted distributions to create separability
    v_fraud = np.random.randn(n_fraud, 28)
    # Apply different shifts to different features (mimics real PCA patterns)
    fraud_shifts = np.array([
        -2.5, 2.8, -3.0, 2.2, -1.5, -1.8, -3.2, 0.5, -2.0, -3.5,
        3.0, -2.0, -1.0, -4.0, 0.8, -2.5, -4.5, -1.5, 1.2, 0.5,
        0.8, -0.5, -0.3, 0.2, 0.5, -0.3, 0.8, -0.5
    ])
    v_fraud += fraud_shifts * np.random.uniform(0.3, 1.0, (n_fraud, 28))
    
    # Add some overlap to make it realistic (not perfectly separable)
    noise_factor = 0.6
    v_fraud += np.random.randn(n_fraud, 28) * noise_factor
    
    # ---- Generate Amount feature ----
    # Legitimate: mostly small transactions with some large ones
    amount_legit = np.abs(np.random.exponential(scale=88.0, size=n_legit))
    amount_legit = np.clip(amount_legit, 0, 25691)
    
    # Fraudulent: different distribution (often larger or very specific amounts)
    amount_fraud = np.concatenate([
        np.abs(np.random.exponential(scale=150.0, size=n_fraud // 2)),
        np.random.uniform(1, 500, n_fraud - n_fraud // 2)
    ])
    np.random.shuffle(amount_fraud)
    amount_fraud = np.clip(amount_fraud, 0, 25691)
    
    # ---- Combine into DataFrame ----
    # Legitimate transactions
    legit_data = np.column_stack([
        time_legit,
        v_legit,
        amount_legit,
        np.zeros(n_legit)
    ])
    
    # Fraudulent transactions
    fraud_data = np.column_stack([
        time_fraud,
        v_fraud,
        amount_fraud,
        np.ones(n_fraud)
    ])
    
    # Combine and shuffle
    all_data = np.vstack([legit_data, fraud_data])
    np.random.shuffle(all_data)
    
    # Sort by time (as in the real dataset)
    all_data = all_data[all_data[:, 0].argsort()]
    
    # Create column names
    columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    
    df = pd.DataFrame(all_data, columns=columns)
    df["Class"] = df["Class"].astype(int)
    
    # Round values for cleaner output
    df["Time"] = df["Time"].round(1)
    df["Amount"] = df["Amount"].round(2)
    for col in [f"V{i}" for i in range(1, 29)]:
        df[col] = df[col].round(8)
    
    return df


def save_dataset(df, path=DATASET_PATH):
    """Save the dataset to CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"\n  Dataset saved to: {path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Shape: {df.shape}")


def print_dataset_summary(df):
    """Print a summary of the generated dataset."""
    print(f"\n{'='*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"\n  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\n  Class Distribution:")
    for cls, count in df["Class"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        label = "Legitimate" if cls == 0 else "Fraud"
        print(f"    {label} (Class {cls}): {count:,} ({pct:.3f}%)")
    
    print(f"\n  Feature Statistics:")
    print(f"    Time range: {df['Time'].min():.0f} - {df['Time'].max():.0f} seconds")
    print(f"    Amount range: ${df['Amount'].min():.2f} - ${df['Amount'].max():.2f}")
    print(f"    Amount mean: ${df['Amount'].mean():.2f}")
    print(f"    Amount median: ${df['Amount'].median():.2f}")
    
    print(f"\n  Missing values: {df.isnull().sum().sum()}")
    print(f"  Duplicate rows: {df.duplicated().sum()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("\n" + "🔒 Credit Card Fraud Detection - Dataset Generator".center(60))
    print("=" * 60)
    
    if os.path.exists(DATASET_PATH):
        print(f"\n  ⚠️  Dataset already exists at: {DATASET_PATH}")
        response = input("  Overwrite? (y/n): ").strip().lower()
        if response != 'y':
            print("  Aborted.")
            sys.exit(0)
    
    # Generate dataset
    df = generate_synthetic_dataset()
    
    # Print summary
    print_dataset_summary(df)
    
    # Save to CSV
    save_dataset(df)
    
    print(f"\n  ✅ Dataset generation complete!")
    print(f"  📁 You can now run: python main.py")
    print(f"{'='*60}\n")
