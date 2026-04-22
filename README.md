# 🔒 Credit Card Fraud Detection System

## Real-Time Credit Card Fraud Detection Using Advanced Data Science Techniques

---

### 👥 Team Members
| Name | Role |
|------|------|
| **Allah Yar Durrani** | Team Lead / Feature Engineering |
| **Muskan Khan** | Data Preprocessing / Visualization |
| **M. Abdullah** | Model Training / Evaluation |
| **Mujtaba Khan** | Documentation / Presentation |

---

## 📋 Project Overview

This project builds an **end-to-end machine learning pipeline** to detect fraudulent credit card transactions in real-time. Using anonymized transaction data, the system identifies suspicious patterns amidst heavily imbalanced datasets (fraud < 1% of transactions), demonstrating advanced data science skills relevant to banking cybersecurity.

### Key Challenges
- **Extreme class imbalance**: Only ~0.17% of transactions are fraudulent
- **High-dimensional PCA features**: V1-V28 are anonymized via PCA
- **Real-time detection needs**: Models must be fast and accurate
- **Business impact**: False negatives (missed fraud) are extremely costly

---

## 🏗️ Project Architecture

```
credit-card-fraud-detection/
│
├── main.py                    # 🚀 Main pipeline (run this!)
├── dashboard.py               # 🛡️ Real-Time Monitoring Dashboard
├── config.py                  # ⚙️  Configuration settings
├── generate_dataset.py        # 📊 Synthetic dataset generator
├── requirements.txt           # 📦 Dependencies
├── README.md                  # 📖 This file
│
├── src/                       # 📂 Source modules
│   ├── __init__.py
│   ├── data_loader.py         # Data loading & EDA
│   ├── preprocessing.py       # Data cleaning & scaling
│   ├── feature_engineering.py # Feature creation
│   ├── visualization.py       # All visualizations
│   ├── model_training.py      # Model training with SMOTE
│   ├── evaluation.py          # Model evaluation & reports
│   └── inference.py          # 🧠 Real-time prediction engine
│
├── data/                      # 📁 Dataset directory
│   ├── creditcard.csv         # Dataset (generated or from Kaggle)
│   └── simulation_test_set.csv# 🧪 Simulation data for dashboard
│
├── models/                    # 💾 Saved trained models
│   ├── random_forest_model.pkl
│   ├── scalers.pkl            # 📏 Scaling parameters for inference
│   └── ... 
│
└── reports/                   # 📄 Generated reports & figures
    ├── evaluation_report.txt
    ├── evaluation_results.json
    └── figures/
        ├── 01_class_distribution.png
        ├── 02_amount_distribution.png
        ├── 03_time_distribution.png
        ├── 04_correlation_heatmap.png
        ├── 05_feature_distributions.png
        ├── 06_confusion_matrix_*.png
        ├── 07_roc_curves.png
        ├── 08_precision_recall_curves.png
        ├── 09_model_comparison.png
        ├── 10_feature_importance_*.png
        └── 11_threshold_analysis_*.pn
```

---

---

## 📊 About the Dataset (Kaggle)

This project uses the **real-world Credit Card Fraud Detection dataset** provided by the Machine Learning Group at ULB (Université Libre de Bruxelles).

- **Source**: [mlg-ulb via Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Timeframe**: Transactions made in September 2013 by European cardholders.
- **Privacy**: Features **V1 to V28** are numerical result of a PCA transformation. Due to confidentiality, original features (like merchant name, location, user age) are not available.
- **Target**: The **Class** column (0 = Legitimate, 1 = Fraud).

---

## 🚀 Installation & Step-by-Step Guide

To use the **Real-Time Monitor** and **Batch Processing**, follow these steps in order:

### 1. 📦 Setup Environment
Create a dedicated virtual environment for the project to keep dependencies clean:
```bash
# Create the virtual environment
python -m venv venv

# Activate it (Windows)
.\venv\Scripts\activate

# Activate it (macOS/Linux)
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt pytest fastapi uvicorn shap
```

### 🧠 2. Training the Models (Crucial)
You must run the training pipeline at least once to generate the **saved model files** (`.pkl`) and the **scalers**.

*   **Fast Mode (Recommended)**: Trains only the primary Random Forest model for quick results.
    ```bash
    python main.py --fast --skip-eda
    ```
*   **Full Mode**: Trains all 5 models (Logistic Regression, RF, Gradient Boosting, DT, KNN).
    ```bash
    python main.py
    ```

### 🛡️ 3. Launch the Real-Time Dashboard
Once training is done, open the UI where you can check "running cards" and upload your own files:
```bash
streamlit run dashboard.py
```

### 🔌 4. Integration via REST API (New)
For production systems, you can now call the fraud detector via a clean REST API:
```bash
python api.py
```
*Port: 8000. Endpoints: `/health`, `/predict`*

### 🧪 5. Running Unit Tests
Ensure the system is working correctly by running the test suite:
```bash
python -m pytest tests/
```

---

## 📤 Using Your Own Data (Batch Processing)

Once the dashboard is open:
1. Navigate to the **"Batch Processing"** tab.
2. Upload a CSV file. It **MUST** have the same columns as the Kaggle dataset (`Time`, `V1-V28`, and `Amount`).
3. Click **"Run Fraud Analysis"**.
4. Download the final report with the **Legitimate/Fraud** predictions added to each row.

---

---

## 📊 Pipeline Steps

### Step 1: Data Loading & Exploration
- Load the credit card transaction dataset (284,807 transactions)
- Perform comprehensive EDA (missing values, duplicates, statistics)
- Analyze class distribution (99.83% legitimate, 0.17% fraud)

### Step 2: Data Preprocessing
- **Quality checks**: Missing values, infinite values, outliers
- **Feature scaling**: RobustScaler for Amount and Time (handles outliers)
- **Train-test split**: 80/20 stratified split to maintain class ratios

### Step 3: Feature Engineering
- **Time features**: Hour extraction, cyclical encoding (sin/cos), night flag
- **Amount features**: Log transform, binning, z-score, high-value flag
- **PCA interactions**: Magnitude, mean, std, key feature products

### Step 4: Model Training with SMOTE
- **SMOTE** (Synthetic Minority Over-sampling Technique): Generates synthetic fraud samples
- **5 Models trained**:
  - Logistic Regression (with balanced class weights)
  - Random Forest (ensemble of decision trees)
  - Gradient Boosting (sequential boosting)
  - Decision Tree (interpretable baseline)
  - K-Nearest Neighbors (distance-based)
- **5-Fold Stratified Cross-Validation** for reliable estimates

### Step 5: Model Evaluation
- **11+ metrics** per model including:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC, Average Precision (PR-AUC)
  - Matthews Correlation Coefficient, Cohen's Kappa
  - False Positive Rate, False Negative Rate
- **Business impact analysis** (estimated savings/losses)
- **Threshold optimization** for best F1-Score

---

## 📈 Visualizations Generated

| # | Visualization | Purpose |
|---|---------------|---------|
| 1 | Class Distribution | Shows extreme class imbalance |
| 2 | Amount Distribution | Transaction amounts by class |
| 3 | Time Distribution | Temporal patterns in fraud |
| 4 | Correlation Heatmap | Feature relationships |
| 5 | Feature Distributions | V-feature overlaps by class |
| 6 | Confusion Matrices | TP/TN/FP/FN for each model |
| 7 | ROC Curves | Model discrimination ability |
| 8 | Precision-Recall Curves | Critical for imbalanced data |
| 9 | Model Comparison | Dashboard with radar chart |
| 10 | Feature Importance | Top predictive features |
| 11 | Threshold Analysis | Optimal classification threshold |

---

## 🔑 Key Concepts Demonstrated

### Why Accuracy Fails for Fraud Detection
With 99.83% legitimate transactions, a model that **always predicts "legitimate"** achieves 99.83% accuracy but catches **zero fraud**. This is why we use:
- **Recall**: How many actual frauds did we catch?
- **Precision**: Of flagged transactions, how many were truly fraud?
- **F1-Score**: Harmonic mean balancing precision and recall

### SMOTE for Class Imbalance
Instead of simple oversampling (duplicating fraud cases), SMOTE creates **synthetic** minority samples by interpolating between existing fraud cases and their nearest neighbors, producing more diverse training data.

### Precision-Recall Tradeoff
- **High threshold**: High precision (fewer false alarms) but low recall (miss more fraud)
- **Low threshold**: High recall (catch more fraud) but low precision (more false alarms)
- **Optimal threshold**: Maximizes F1-Score (balance point)

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|------|---------|
| **Python 3.x** | Primary programming language |
| **Pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **Scikit-learn** | Machine learning algorithms |
| **imbalanced-learn** | SMOTE implementation |
| **Matplotlib** | Static visualizations |
| **Seaborn** | Statistical data visualization |
| **Streamlit** | Real-time monitoring dashboard UI |
| **Plotly** | Interactive risk charts |
| **Joblib** | Model & Scaler serialization |
| **VS Code** | Development environment |

---

## 📊 Dataset Details

The dataset contains credit card transactions made by European cardholders in September 2013.

| Feature | Description |
|---------|-------------|
| **Time** | Seconds elapsed since first transaction |
| **V1-V28** | PCA-transformed features (anonymized) |
| **Amount** | Transaction amount |
| **Class** | 0 = Legitimate, 1 = Fraud |

- **Total transactions**: 284,807
- **Fraud rate**: 0.172% (492 frauds)
- **Total features**: 31

---

## 📄 Output Reports

After running the pipeline, check the `reports/` directory for:
- `evaluation_report.txt` - Comprehensive text report with all metrics
- `evaluation_results.json` - Machine-readable results
- `figures/` - All generated visualization PNG files

---

## 📚 References

1. [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813)
3. [Scikit-learn Documentation](https://scikit-learn.org/stable/)
4. [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)

---

## 📝 License

This project is developed for educational purposes as part of a Data Science course project.

---

*Developed with ❤️ by Team Durrani-Muskan-Abdullah-Mujtaba*
#   d a t a s c i e n c e _ p r o j e c t  
 