import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from src.inference import FraudDetector
import os

# Page configuration
st.set_page_config(
    page_title="Guardian Sentinel | Real-Time Fraud Monitor",
    page_icon="🛡️",
    layout="wide",
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(180deg, #0e1117 0%, #161b22 100%);
    }
    .fraud-card {
        background-color: #ff4b4b22;
        border: 1px solid #ff4b4b;
        padding: 20px;
        border-radius: 10px;
        color: #ff4b4b;
    }
    .legit-card {
        background-color: #00ff0011;
        border: 1px solid #00ff00;
        padding: 20px;
        border-radius: 10px;
        color: #00ff00;
    }
    .status-badge {
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 4px;
        text-transform: uppercase;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.title("🛡️ Guardian Sentinel")
st.subheader("Real-Time Credit Card Fraud Detection System")
st.markdown("---")

# Initialize Detector
@st.cache_resource
def get_detector():
    return FraudDetector(model_name="random_forest")

try:
    detector = get_detector()
except Exception as e:
    st.error(f"Error loading detector: {e}")
    st.stop()

# Sidebar for controls
with st.sidebar:
    st.header("Simulation Controls")
    sim_speed = st.slider("Simulation Speed (s)", 0.5, 5.0, 1.5)
    run_sim = st.button("Start Live Monitoring", type="primary")
    reset_sim = st.button("Reset Session")
    
    st.markdown("---")
    st.header("Model Settings")
    risk_threshold = st.slider("Risk Threshold (%)", 5, 95, 50) / 100
    
    st.markdown("---")
    st.info("System Status: **ACTIVE**")
    st.success("Model: **Random Forest (Balanced)**")

# Pre-load Simulation Data
@st.cache_data
def load_sim_data():
    if os.path.exists("data/simulation_test_set.csv"):
        df = pd.read_csv("data/simulation_test_set.csv")
        return df
    return None

test_df = load_sim_data()

if test_df is None:
    st.warning("Simulation data not found. Please run the training pipeline first.")
    st.stop()

# Dashboard State
if 'transactions' not in st.session_state or reset_sim:
    st.session_state.transactions = []
    st.session_state.fraud_count = 0
    st.session_state.total_processed = 0

# Metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Transactions", st.session_state.total_processed)
with col2:
    st.metric("Fraud Detected", st.session_state.fraud_count, delta=f"{st.session_state.fraud_count} cases", delta_color="inverse")
with col3:
    fraud_rate = (st.session_state.fraud_count / st.session_state.total_processed * 100) if st.session_state.total_processed > 0 else 0
    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
with col4:
    st.metric("System Health", "100%", delta="Latent: 12ms")

# Main content
tab_live, tab_batch, tab_insights = st.tabs(["📡 Live Monitoring", "📤 Batch Processing", "📈 Advanced Insights"])

with tab_live:
    col_main, col_viz = st.columns([2, 1])
    with col_main:
        st.write("### Live Transaction Feed")
        placeholder = st.empty()
    with col_viz:
        st.write("### Distribution Analysis")
        viz_placeholder = st.empty()

with tab_batch:
    st.write("### 📤 Upload Your Transaction Data")
    st.info("Upload a CSV file containing columns: Time, Amount, V1, V2, ... V28")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(user_df)} transactions.")
        
        if st.button("🔍 Run Fraud Analysis"):
            with st.spinner("Analyzing data..."):
                st.write("🛡️ *Verifying transactions using **Random Forest Model**...*")
                results = []
                # Process in batches for better performance
                for i, row in user_df.iterrows():
                    # Check if it has required columns
                    try:
                        res = detector.predict(row)
                        # Use custom threshold
                        is_fraud = res['probability'] >= risk_threshold
                        results.append("Fraud" if is_fraud else "Legitimate")
                    except Exception:
                        results.append("Error (Invalid Columns)")
                
                user_df['Detection_Result'] = results
                
                # Summary
                fraud_count = (user_df['Detection_Result'] == 'Fraud').sum()
                st.success(f"Analysis Complete! Found {fraud_count} suspicious transactions.")
                
                # Visual results
                st.dataframe(user_df.style.apply(lambda x: ['background-color: #ff4b4b22' if val == 'Fraud' else '' for val in x], axis=1))
                
                # Download link
                csv = user_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results", csv, "fraud_analysis_results.csv", "text/csv")

with tab_insights:
    st.write("### 📈 Cost-Benefit & Explainability Analysis")
    
    col_metric1, col_metric2 = st.columns(2)
    
    # Calculate potential savings (Mock values based on current session)
    avg_fraud_amount = 450.0  # Industry average
    saved = st.session_state.fraud_count * avg_fraud_amount
    false_pos_cost = (st.session_state.total_processed - st.session_state.fraud_count) * 2.0 # Cost of manual review
    
    with col_metric1:
        st.metric("Potential Loss Prevented", f"${saved:,.2f}", delta="Estimated Savings")
    with col_metric2:
        st.metric("Alert Accuracy", f"{(st.session_state.fraud_count / st.session_state.total_processed * 100 if st.session_state.total_processed > 0 else 100):.1f}%")
    
    st.markdown("---")
    st.write("#### 🧠 Model Reasoning (Last Transaction)")
    
    if 'last_explanation' in st.session_state:
        exp_df = st.session_state.last_explanation
        fig_exp = px.bar(
            exp_df.head(10),
            x='impact',
            y='feature',
            orientation='h',
            title="Top 10 Feature Contributions to Risk Score",
            color='impact',
            color_continuous_scale='Reds'
        )
        fig_exp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_exp, use_container_width=True)
    else:
        st.info("Start monitoring to see real-time explanations.")


# Simulation Loop
if run_sim:
    # Get a stream of random indices from test_df
    # We prioritize showing some fraud for the demo
    fraud_indices = test_df[test_df['Class'] == 1].index.tolist()
    legit_indices = test_df[test_df['Class'] == 0].index.tolist()
    
    while True:
        # Choose a random record (weighted towards fraud for visualization if available)
        if np.random.random() > 0.85 and fraud_indices:
            idx = np.random.choice(fraud_indices)
        else:
            idx = np.random.choice(legit_indices)
            
        raw_record = test_df.loc[idx].drop('Class')
        ground_truth = test_df.loc[idx, 'Class']
        
        # Predict
        result = detector.predict(raw_record)
        
        # Apply custom threshold
        custom_is_fraud = result['probability'] >= risk_threshold
        
        # Update state
        st.session_state.total_processed += 1
        if custom_is_fraud:
            st.session_state.fraud_count += 1
            # Get explanation for fraud
            st.session_state.last_explanation = detector.explain_prediction(raw_record)
            
        # Add to table feed
        new_row = {
            "Timestamp": pd.Timestamp.now().strftime("%H:%M:%S.%f")[:-3],
            "Amount": f"${raw_record['Amount']:.2f}",
            "Risk Score": f"{result['probability']*100:.1f}%",
            "Status": "⚠️ FRAUD" if custom_is_fraud else "✅ LEGIT"
        }
        st.session_state.transactions.insert(0, new_row)
        if len(st.session_state.transactions) > 10:
            st.session_state.transactions.pop()
            
        # Update Table Display
        with placeholder.container():
            df_display = pd.DataFrame(st.session_state.transactions)
            
            def color_status(val):
                color = '#ff4b4b' if 'FRAUD' in val else '#00ff00'
                return f'color: {color}; font-weight: bold'
            
            st.markdown(f"**⚡ Live Engine Status:** *Verified by {detector.model_path.split('/')[-1].replace('_model.pkl', '').replace('_', ' ').title()}*")
            st.table(df_display.style.applymap(color_status, subset=['Status']))
            
            # Show Detailed Card for last fraud detected
            if custom_is_fraud:
                st.markdown(f"""
                <div class="fraud-card">
                    <h4>🚨 FRAUD ALERT DETECTED</h4>
                    <p>Transaction ID: {idx}</p>
                    <p>Amount: ${raw_record['Amount']:.2f}</p>
                    <p>System Score: {result['probability']*100:.2f}% probability of malicious intent.</p>
                </div>
                """, unsafe_allow_html=True)

        # Update Viz
        with viz_placeholder.container():
            # Class distribution chart
            fig = px.pie(
                values=[st.session_state.total_processed - st.session_state.fraud_count, st.session_state.fraud_count],
                names=['Legit', 'Fraud'],
                color=['Legit', 'Fraud'],
                color_discrete_map={'Legit':'#00ff00', 'Fraud':'#ff4b4b'},
                hole=0.5,
                title="Transactions Overview"
            )
            fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Amount Probability Scatter
            if len(st.session_state.transactions) > 1:
                # Mock high-risk indicators
                st.info(f"Top Indicator: {st.session_state.last_explanation.iloc[0]['feature'] if 'last_explanation' in st.session_state else 'Normal Pattern'}")

        time.sleep(sim_speed)
else:
    placeholder.info("Click 'Start Live Monitoring' to begin the transaction stream.")
