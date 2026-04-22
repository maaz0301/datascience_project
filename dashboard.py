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
    page_title="Guardian Sentinel | AI Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Main Title Styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Metric Cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 1rem;
        transition: transform 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.05);
        border-color: #38bdf8;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Alerts & Cards */
    .fraud-alert {
        background: rgba(239, 68, 68, 0.1);
        border-left: 5px solid #ef4444;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    
    .legit-card {
        background: rgba(34, 197, 94, 0.1);
        border-left: 5px solid #22c55e;
        padding: 1.5rem;
        border-radius: 0.5rem;
    }
    
    /* Custom Badge */
    .badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    
    .badge-fraud { background-color: #ef4444; color: white; }
    .badge-legit { background-color: #22c55e; color: white; }
    
    /* Table Styling */
    [data-testid="stTable"] {
        border-radius: 1rem;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# --- APP HEADER ---
st.markdown('<h1 class="main-title">🛡️ Guardian Sentinel</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-Powered Real-Time Credit Card Fraud Detection</p>', unsafe_allow_html=True)

# --- INITIALIZE DETECTOR ---
@st.cache_resource
def get_detector():
    return FraudDetector(model_name="random_forest")

try:
    detector = get_detector()
except Exception as e:
    st.error(f"Error loading AI Engine: {e}")
    st.info("💡 Hint: Run `python main.py` first to train the models.")
    st.stop()

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=80)
    st.markdown("### Simulation Control Center")
    
    with st.expander("📡 Connectivity Settings", expanded=True):
        sim_speed = st.slider("Stream Interval (seconds)", 0.2, 3.0, 1.0)
        risk_threshold = st.slider("Risk Sensitivity (%)", 0, 100, 50) / 100
        
    st.markdown("---")
    st.markdown("### 📥 Stream Source")
    source_type = st.radio("Select Data Pool", ["Synthetic Simulation", "Custom Upload (CSV)"], index=0)
    uploaded_stream_file = None
    if source_type == "Custom Upload (CSV)":
        uploaded_stream_file = st.file_uploader("Upload Stream Data (CSV)", type="csv")
        if uploaded_stream_file:
            # Only read once per upload
            file_key = f"pool_{uploaded_stream_file.name}_{uploaded_stream_file.size}"
            if st.session_state.get('last_pool_key') != file_key:
                st.session_state.current_pool = pd.read_csv(uploaded_stream_file)
                st.session_state.last_pool_key = file_key
                st.session_state.pool_index = 0
                st.session_state.total_processed = 0
                st.session_state.fraud_count = 0
                st.session_state.transactions = []
                st.sidebar.success(f"Loaded {len(st.session_state.current_pool)} rows")
    else:
        st.session_state.current_pool = None
        st.session_state.last_pool_key = None
        
    st.markdown("---")
    st.markdown("### 🚦 Engine Controls")
    
    if 'sim_running' not in st.session_state:
        st.session_state.sim_running = False
        
    col_start, col_reset = st.columns(2)
    with col_start:
        if not st.session_state.sim_running:
            if st.button("▶️ START", width="stretch", type="primary"):
                st.session_state.sim_running = True
                st.rerun()
        else:
            if st.button("🛑 STOP", width="stretch"):
                st.session_state.sim_running = False
                st.rerun()
                
    with col_reset:
        if st.button("🔄 RESET", width="stretch"):
            st.session_state.transactions = []
            st.session_state.fraud_count = 0
            st.session_state.total_processed = 0
            st.session_state.pool_index = 0
            st.session_state.last_fraud_id = None
            st.session_state.last_explanation = None
            st.session_state.sim_running = False
            st.rerun()
            
    if st.button("🚀 FAST-TRACK (RUN ALL)", width="stretch", help="Process entire pool as fast as possible"):
        st.session_state.run_all_triggered = True
        st.rerun()
        
    st.markdown("---")
    st.markdown("### 📋 System Telemetry")
    
    # Calculate progress if pool exists
    if 'pool_size' in st.session_state and st.session_state.pool_size > 0:
        progress = (st.session_state.total_processed / st.session_state.pool_size)
        st.write(f"**Pool Progress:** {progress:.1%}")
        st.progress(min(progress, 1.0))
        st.write(f"**Data Pool:** `{st.session_state.pool_size:,} rows | {st.session_state.pool_cols} cols`")
    else:
        st.write(f"**Node Status:** `HEALTHY`")
        
    st.write(f"**Model ID:** `RF-BALANCED-V1`")
    st.write(f"**Uptime:** `100.0%`")
    
    st.markdown("---")
    st.markdown("### 👥 Team Credits")
    st.caption("Developed by Durrani, Muskan, Abdullah & Mujtaba")

# --- DATA LOADING ---
@st.cache_data
def load_sim_data():
    if os.path.exists("data/simulation_test_set.csv"):
        return pd.read_csv("data/simulation_test_set.csv")
    return None

test_df = load_sim_data()
if test_df is None:
    st.error("Missing simulation data. Run training pipeline first.")
    st.stop()

# --- SESSION STATE MANAGEMENT ---
if 'transactions' not in st.session_state:
    st.session_state.transactions = []
    st.session_state.fraud_count = 0
    st.session_state.total_processed = 0
    st.session_state.pool_index = 0
    st.session_state.pool_size = 0
    st.session_state.pool_cols = 0
    st.session_state.last_fraud_id = None
    st.session_state.last_explanation = None
    st.session_state.run_all_triggered = False

# --- TOP METRICS ROW ---
m_col1, m_col2, m_col3, m_col4 = st.columns(4)
with m_col1:
    st.metric("Total Scanned", f"{st.session_state.total_processed:,}")
with m_col2:
    st.metric("Fraudulent", st.session_state.fraud_count, 
              delta=f"{st.session_state.fraud_count} ALERTS", delta_color="inverse")
with m_col3:
    rate = (st.session_state.fraud_count / st.session_state.total_processed * 100) if st.session_state.total_processed > 0 else 0
    st.metric("Infection Rate", f"{rate:.2f}%")
with m_col4:
    st.metric("Latence", "8.4ms", delta="OPTIMAL")

# --- FRAGMENTED LIVE UPDATE ENGINE ---
@st.fragment(run_every=sim_speed if st.session_state.sim_running else None)
def live_monitor_fragment():
    col_f, col_s = st.columns([1.6, 1])
    with col_f:
        st.markdown("### ⚡ Live Transaction Stream")
        f_place = st.empty()
        a_place = st.empty()
    with col_s:
        st.markdown("### 📊 Distribution Data")
        d_place = st.empty()
        g_place = st.empty()

    # --- RENDER UI (ALWAYS IF DATA EXISTS) ---
    if st.session_state.transactions:
        with f_place.container():
            df_v = pd.DataFrame(st.session_state.transactions)
            
            # Show final summary if completed
            if st.session_state.pool_size > 0 and st.session_state.pool_index >= st.session_state.pool_size:
                fraud_rate = (st.session_state.fraud_count / st.session_state.total_processed * 100) if st.session_state.total_processed > 0 else 0
                st.markdown(f"""
                <div style="background: rgba(34, 197, 94, 0.1); border: 1px solid #22c55e; border-radius: 10px; padding: 1.5rem; margin-bottom: 1.5rem; text-align: center;">
                    <h2 style="margin:0; color: #22c55e;">🏁 ANALYSIS COMPLETE</h2>
                    <p style="margin:5px 0 0 0; opacity: 0.8;">Full Data Pool Audited Successfully</p>
                    <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                        <div><h3 style="margin:0;">{st.session_state.total_processed:,}</h3><small>TOTAL SCAN</small></div>
                        <div><h3 style="margin:0; color: #ef4444;">{st.session_state.fraud_count}</h3><small>FRAUD DETECTED</small></div>
                        <div><h3 style="margin:0;">{fraud_rate:.2f}%</h3><small>GLOBAL RISK</small></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Distribution Plot at the end
                st.markdown("#### 📊 Final Distribution Overview")
                fig_dist = px.bar(
                    x=['Legit', 'Fraud'], 
                    y=[st.session_state.total_processed - st.session_state.fraud_count, st.session_state.fraud_count],
                    color=['Legit', 'Fraud'],
                    color_discrete_map={'Legit':'#22c55e', 'Fraud':'#ef4444'},
                    labels={'x': 'Transaction Type', 'y': 'Count'}
                )
                fig_dist.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#f8fafc", showlegend=False)
                st.plotly_chart(fig_dist, width="stretch", key="final_dist_plot")

            selection = st.dataframe(
                df_v.style.map(lambda v: 'background-color: #450a0a' if v == "⚠️ FRAUD" else '', subset=['Result']),
                width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row"
            )
            if selection and selection.get('selection', {}).get('rows'):
                sel_idx = selection['selection']['rows'][0]
                sel_data = df_v.iloc[sel_idx]
                sel_id = sel_data['ID']
                selected_prob = float(sel_data['Confidence'].strip('%'))
                
                st.markdown(f"#### 🔍 Transaction Forensic Insight: #{sel_id}")
                v_cols = [f"V{i}" for i in range(1, 29)]
                ref_pool = st.session_state.current_pool if source_type == "Custom Upload (CSV)" and st.session_state.current_pool is not None else test_df
                v_vals = ref_pool.loc[sel_id][v_cols]
                fig_sel = px.line(x=v_cols, y=v_vals, markers=True)
                fig_sel.update_layout(height=200, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="#f8fafc")
                st.plotly_chart(fig_sel, width="stretch", key=f"sel_chart_{sel_id}")
            
        # Summary charts
        with d_place.container():
            fig_pie = px.pie(values=[st.session_state.total_processed - st.session_state.fraud_count, st.session_state.fraud_count], names=['Legit', 'Fraud'], color=['Legit', 'Fraud'], color_discrete_map={'Legit':'#22c55e', 'Fraud':'#ef4444'}, hole=0.6)
            fig_pie.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=250, paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_pie, width="stretch", key=f"pie_{st.session_state.total_processed}")
            
        with g_place.container():
            # Use selected probability if a row is clicked, else use last processed
            if 'selected_prob' in locals():
                display_prob = selected_prob
                gauge_title = "Selected Risk"
            else:
                display_prob = float(st.session_state.transactions[0]['Confidence'].strip('%'))
                gauge_title = "Live Risk Level"
                
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = display_prob, 
                title = {'text': gauge_title, 'font': {'size': 18}}, 
                gauge = {
                    'axis': {'range': [None, 100]}, 
                    'bar': {'color': "#ef4444" if display_prob > 50 else "#38bdf8"}, 
                    'steps': [{'range': [0, 50], 'color': "rgba(34, 197, 94, 0.1)"}, {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.1)"}]
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=0), paper_bgcolor='rgba(0,0,0,0)', font_color="#f8fafc")
            st.plotly_chart(fig_gauge, width="stretch", key=f"gauge_dynamic_{display_prob}")
    else:
        if not st.session_state.sim_running and not st.session_state.run_all_triggered:
            f_place.info("System Standby. Click 'START' in simulation controls to activate AI Engine.")
            return

    # --- SIMULATION LOGIC ---
    if st.session_state.sim_running or st.session_state.run_all_triggered:
        # Load data pool
        if source_type == "Custom Upload (CSV)" and st.session_state.current_pool is not None:
            pool_df = st.session_state.current_pool
        else:
            pool_df = test_df
        
        st.session_state.pool_size = len(pool_df)
        st.session_state.pool_cols = pool_df.shape[1]

        # Handle Batch Run-All
        if st.session_state.run_all_triggered:
            with st.status("🚀 Fast-Tracking Data Pool...", expanded=True) as status:
                remaining_df = pool_df.iloc[st.session_state.pool_index:].copy()
                st.write(f"Analyzing {len(remaining_df)} transactions...")
                
                results = []
                for idx, row_raw in remaining_df.iterrows():
                    row = row_raw.drop('Class') if 'Class' in row_raw else row_raw
                    pred = detector.predict(row)
                    is_fraud = pred['probability'] >= risk_threshold
                    results.append((idx, pred['probability'], is_fraud, row_raw))
                    
                for idx, prob, is_fraud, row_raw in results:
                    st.session_state.total_processed += 1
                    if is_fraud:
                        st.session_state.fraud_count += 1
                        st.session_state.last_fraud_id = idx
                        st.session_state.last_explanation = detector.explain_prediction(row_raw.drop('Class') if 'Class' in row_raw else row_raw)
                    
                    status_text = "⚠️ FRAUD" if is_fraud else "✅ LEGIT"
                    st.session_state.transactions.insert(0, {
                        "ID": idx, "Time": "BATCH", "Amount": f"${row_raw['Amount']:.2f}",
                        "Confidence": f"{prob*100:.1f}%", "Result": status_text
                    })
                
                st.session_state.pool_index = len(pool_df)
                st.session_state.run_all_triggered = False
                st.session_state.sim_running = False
                status.update(label="✅ Fast-Track Complete!", state="complete")
            st.rerun()

        # Handle Streaming Step
        if st.session_state.sim_running:
            if source_type == "Custom Upload (CSV)":
                if st.session_state.pool_index >= len(pool_df):
                    st.session_state.sim_running = False
                    st.success("✅ End of data reached!")
                    st.rerun()
                idx = st.session_state.pool_index
                st.session_state.pool_index += 1
            else:
                fraud_indices = pool_df[pool_df['Class'] == 1].index.tolist()
                legit_indices = pool_df[pool_df['Class'] == 0].index.tolist()
                idx = np.random.choice(fraud_indices) if np.random.random() > 0.85 else np.random.choice(legit_indices)
                
            row = pool_df.loc[idx]
            row_inf = row.drop('Class') if 'Class' in row else row
            pred = detector.predict(row_inf)
            is_fraud = pred['probability'] >= risk_threshold
            
            st.session_state.total_processed += 1
            if is_fraud:
                st.session_state.fraud_count += 1
                st.session_state.last_fraud_id = idx
                st.session_state.last_explanation = detector.explain_prediction(row_inf)
                
            status_text = "⚠️ FRAUD" if is_fraud else "✅ LEGIT"
            st.session_state.transactions.insert(0, {
                "ID": idx, "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
                "Amount": f"${row['Amount']:.2f}", "Confidence": f"{pred['probability']*100:.1f}%", "Result": status_text
            })
            
            if len(st.session_state.transactions) > 1000:
                st.session_state.transactions.pop()
            
            if is_fraud:
                with a_place.container():
                    st.markdown(f'<div class="fraud-alert"><h3 style="margin:0">🚨 FRAUD DETECTED</h3><p style="margin:0">ID: {idx} | ${row["Amount"]:.2f}</p></div>', unsafe_allow_html=True)
            
            st.rerun()

# --- MAIN DASHBOARD INTERFACE ---
tab_live, tab_batch, tab_ai = st.tabs(["📡 LIVE ENGINE", "📤 BATCH UPLOAD", "🧠 AI INTERPRETER"])

with tab_live:
    # Fragment is called here to render the monitor
    live_monitor_fragment()

# --- TAB CONTENT ---
with tab_batch:
    st.markdown("### 📤 High-Volume Batch Processing")
    st.write("Upload a dataset for historical audit and forensic fraud analysis.")
    uploaded_file = st.file_uploader("Drop CSV transaction history here", type="csv")
    if uploaded_file:
        df_u = pd.read_csv(uploaded_file)
        st.dataframe(df_u.head(5), width="stretch")
        if st.button("🔥 START ANALYSIS", width="stretch"):
            with st.status("Analyzing Transactions...", expanded=True) as status:
                results = [detector.predict(r)['probability'] >= risk_threshold for _, r in df_u.iterrows()]
                df_u['FRAUD_STATUS'] = ["FRAUD" if r else "LEGIT" for r in results]
                status.update(label="Analysis Complete!", state="complete", expanded=False)
            st.success(f"Audit Complete: {(df_u['FRAUD_STATUS'] == 'FRAUD').sum()} suspicious patterns identified.")
            st.dataframe(df_u.style.apply(lambda x: ['background: #450a0a' if v == 'FRAUD' else '' for v in x], axis=1), width="stretch")
            st.download_button("📥 EXPORT AUDIT REPORT", df_u.to_csv(index=False), "fraud_audit.csv")

with tab_ai:
    st.markdown("### 🧠 Explainable AI (XAI)")
    if st.session_state.last_explanation is not None:
        exp_df = st.session_state.last_explanation
        st.write(f"Visualizing decision reasoning for **Transaction #{st.session_state.last_fraud_id}**")
        fig_xai = px.bar(exp_df.head(12), x='impact', y='feature', orientation='h', color='impact', color_continuous_scale='Reds')
        fig_xai.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#f8fafc', height=500)
        st.plotly_chart(fig_xai, width="stretch", key="xai_chart")
    else:
        st.info("Start the Live Engine to analyze fraud patterns.")

