# 🔒 Real-Time Credit Card Fraud Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-v0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end machine learning pipeline for detecting fraudulent credit card transactions in real-time. Built with advanced data science techniques to handle extreme class imbalance and deliver production-ready banking cybersecurity metrics.

---

## 👥 Meet the Team
| Member | Role | Focus Area |
| :--- | :--- | :--- |
| **Allah Yar Durrani** | Team Lead | Feature Engineering & Architecture |
| **Muskan Khan** | Data Scientist | Preprocessing & Visual Analytics |
| **M. Abdullah** | ML Engineer | Model Training & Evaluation |
| **Mujtaba Khan** | Analyst | Documentation & Business Logic |

---

## 📋 Project Overview
Detecting fraud is a high-stakes challenge for financial institutions. With only **0.17%** of transactions being fraudulent, traditional accuracy metrics fail. This project implements a robust system that prioritizes **Recall** and **Precision-Recall AUC** to catch fraud while minimizing false alarms.

### ✨ Key Features
- **🤖 Multi-Model Ensemble**: Logistic Regression, Random Forest, Gradient Boosting, and more.
- **⚖️ Imbalance Mastery**: Advanced SMOTE and balanced class weights to learn from sparse fraud data.
- **🛡️ Real-Time Monitoring**: Interactive Streamlit dashboard for live transaction simulation.
- **🔌 Enterprise Ready**: FastAPI endpoints for seamless integration into existing banking apps.
- **📊 Business Intelligence**: Cost-benefit analysis for each model based on fraud losses vs. investigation costs.

---

## 🏗️ Project Architecture
```text
credit-card-fraud-detection/
├── main.py                    # 🚀 Complete Pipeline Runner
├── dashboard.py               # 🛡️ Real-Time Streamlit UI
├── api.py                     # 🔌 FastAPI Inference Service
├── config.py                  # ⚙️  Global Configurations
├── src/                       # 📂 Core Logic Modules
│   ├── preprocessing.py       # Data Cleaning & Scaling
│   ├── feature_engineering.py # Time/Amount Domain Features
│   ├── model_training.py      # SMOTE & Model Training
│   ├── evaluation.py          # Detailed Metrics & PR-Curves
│   └── inference.py           # Sub-second Detection Engine
└── reports/                   # 📄 Performance Artifacts
    └── figures/               # Professional Visualizations
```

---

## 🚀 Quick Start Guide

### 1️⃣ Installation
```bash
# Clone and enter directory
git clone https://github.com/maaz0301/datascience_project.git
cd datascience_project

# Setup virtual environment
python -m venv venv
source venv/bin/activate  # Or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Train & Evaluate
Run the full data science pipeline to generate models and reports:
```bash
python main.py
```

### 3️⃣ Launch Dashboard
Experience the real-time detection interface:
```bash
streamlit run dashboard.py
```

---

## 🛡️ Dashboard Usage Guide

The **Guardian Sentinel** interface is divided into three specialized operational modules:

### 📡 1. Live Engine (Real-Time Monitoring)
- **Activation**: Click the **▶️ START** button in the sidebar.
- **Function**: Simulates a live stream of global transactions. The AI analyzes each transaction in sub-seconds.
- **Alerts**: Fraudulent transactions trigger a pulsing red alert box and a high-risk gauge reading.
- **Controls**: Adjust the **Stream Interval** to speed up or slow down the feed, and the **Risk Sensitivity** to fine-tune the detection threshold.

### 📤 2. Batch Upload (Forensic Audit)
- **Function**: Designed for processing historical transaction logs or bulk uploads.
- **How to Use**:
    1. Navigate to the **BATCH UPLOAD** tab.
    2. Upload a CSV file (e.g., `batch_to_process.csv`).
    3. Click **🔥 START ANALYSIS**.
- **Output**: Generates a color-coded report identifying all fraudulent patterns found in the file, which can then be exported as a CSV audit report.

### 🧠 3. AI Interpreter (Explainability)
- **Function**: Provides "Glass Box" transparency for AI decisions.
- **Details**: Shows exactly *why* a transaction was flagged as fraud by visualizing the impact of specific features (Time, Amount, V-components) on the risk score.

---

### 4️⃣ Use REST API
Deploy the model as a microservice:
```bash
python api.py
```

---

## 📊 Pipeline Logic

| Stage | Techniques Used |
| :--- | :--- |
| **Data Quality** | Outlier detection, Robust Scaling, Missing value imputation |
| **Engineering** | Cyclical Time Encoding, Log-Amount transforms, PCA Interaction features |
| **Resampling** | **SMOTE** (Synthetic Minority Over-sampling Technique) |
| **Evaluation** | PR-AUC, F1-Score optimization, Business Impact cost analysis |

---

## 📈 Visualizations & Reports
The system automatically generates a suite of professional reports in the `reports/` directory:
- **Precision-Recall Curves**: The gold standard for imbalanced data.
- **Threshold Analysis**: Finds the "sweet spot" between catching fraud and false alarms.
- **Feature Importance**: Understand which behaviors trigger a fraud alert.
- **Radar Charts**: Comparative performance overview of all trained models.

---

## 🔑 Why This Matters
In banking, **missing 1 fraud case** can result in thousands of dollars in losses, while a **false alarm** only costs a few dollars in investigation time. Our system is optimized to favor **Recall** (catching more fraud) while maintaining a sustainable **Precision** level, delivering a higher **Net Benefit** to the institution.

---

## 🛠️ Tech Stack
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, imbalanced-learn, Matplotlib, Seaborn
- **Frameworks**: Streamlit (UI), FastAPI (API)
- **Environment**: VS Code / Git

---

## 📝 License & Credits
Developed as a portfolio project demonstrating advanced ML skills for Fintech careers.
- **Dataset**: [mlg-ulb via Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Team**: Durrani, Muskan, Abdullah, Mujtaba

---
*Developed with ❤️ for Advanced Data Science*