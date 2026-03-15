import sys
sys.path.append(".")

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

st.set_page_config(
    page_title = "FinShield AI — Fraud Dashboard",
    page_icon  = "🛡️",
    layout     = "wide"
)

@st.cache_resource
def get_supabase():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return create_client(url, key)

supabase = get_supabase()

@st.cache_data(ttl=0)
def load_cases():
    response = supabase.table("fraud_cases").select("*").order("id", desc=True).execute()
    if response.data:
        return pd.DataFrame(response.data)
    return pd.DataFrame()

def clean_decision(val):
    val = str(val).upper()
    if "BLOCK"  in val: return "BLOCK"
    elif "ALLOW" in val: return "ALLOW"
    else:                return "REVIEW"

def extract_reasoning(val):
    """Extract LLM reasoning — remove just the decision word, keep explanation"""
    val = str(val)
    # remove markdown bold
    val = val.replace("**", "")
    # if it starts with just BLOCK/ALLOW/REVIEW on first line, skip that line
    lines = val.strip().split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        if line.upper() in ["BLOCK", "ALLOW", "REVIEW", "1.", "2."]:
            continue
        if line.startswith("1.") or line.startswith("2."):
            cleaned.append(line)
        elif line:
            cleaned.append(line)
    return "\n".join(cleaned).strip() if cleaned else val

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🛡️ FinShield AI — Real-Time Fraud Detection Dashboard")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.divider()

df = load_cases()

if df.empty:
    st.warning("No fraud cases found. Run simulate_stream.py to generate data.")
    st.stop()

df["decision_clean"] = df["decision"].apply(clean_decision)

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total   = len(df)
high    = len(df[df["risk_level"] == "HIGH"])
medium  = len(df[df["risk_level"] == "MEDIUM"])
low     = len(df[df["risk_level"] == "LOW"])
blocked = len(df[df["decision_clean"] == "BLOCK"])

col1.metric("Total Transactions", total)
col2.metric("🔴 High Risk",   high,   delta=f"{round(high/total*100,1)}%"   if total else "0%")
col3.metric("🟡 Medium Risk", medium, delta=f"{round(medium/total*100,1)}%" if total else "0%")
col4.metric("🟢 Low Risk",    low,    delta=f"{round(low/total*100,1)}%"    if total else "0%")
col5.metric("🚫 Blocked",     blocked)

st.divider()

# ── CHARTS ────────────────────────────────────────────────────────────────────
colors = {"HIGH": "#E24B4A", "MEDIUM": "#BA7517", "LOW": "#1D9E75"}

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Risk Level Distribution")
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    fig = px.pie(risk_counts, names="Risk Level", values="Count",
                 color="Risk Level", color_discrete_map=colors)
    fig.update_layout(margin=dict(t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Decision Distribution")
    dec_counts = df["decision_clean"].value_counts().reset_index()
    dec_counts.columns = ["Decision", "Count"]
    dec_colors = {"BLOCK": "#E24B4A", "REVIEW": "#BA7517", "ALLOW": "#1D9E75"}
    fig2 = px.bar(dec_counts, x="Decision", y="Count",
                  color="Decision", color_discrete_map=dec_colors)
    fig2.update_layout(margin=dict(t=0, b=0), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

col_left2, col_right2 = st.columns(2)

with col_left2:
    st.subheader("Risk Score Over Time")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_sorted = df.sort_values("timestamp")
    fig3 = px.line(df_sorted, x="timestamp", y="risk_score",
                   color_discrete_sequence=["#534AB7"])
    fig3.update_layout(xaxis_title="Time", yaxis_title="Risk Score", margin=dict(t=0, b=0))
    st.plotly_chart(fig3, use_container_width=True)

with col_right2:
    st.subheader("Fraud Probability Distribution")
    fig4 = px.histogram(df, x="fraud_probability", nbins=20,
                        color_discrete_sequence=["#534AB7"])
    fig4.update_layout(xaxis_title="Fraud Probability", yaxis_title="Count", margin=dict(t=0, b=0))
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── DETECTION SIGNALS ─────────────────────────────────────────────────────────
st.subheader("Detection Engine Results")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🔍 Anomalies",        len(df[df["anomaly"]  == "ANOMALY"]))
c2.metric("👤 Unusual Behavior", len(df[df["behavior"] == "UNUSUAL"]))
c3.metric("🕸️ Graph Fraud",       len(df[df["graph"]    == "FRAUD_NETWORK"]))
c4.metric("⚡ Velocity Flags",    len(df[df["velocity"] == "SUSPICIOUS"]))
c5.metric("📱 Device/IP Flags",   len(df[df["device"]   == "SUSPICIOUS"]))

st.divider()

# ── TRANSACTION TABLE ─────────────────────────────────────────────────────────
st.subheader("Recent Transactions")

cf1, cf2, cf3 = st.columns(3)
with cf1:
    risk_filter = st.selectbox("Filter by Risk", ["All", "HIGH", "MEDIUM", "LOW"])
with cf2:
    min_amount = st.number_input("Min Amount", value=0.0)
with cf3:
    show_n = st.slider("Show last N", 5, 200, 50)

df_f = df.copy()
if risk_filter != "All":
    df_f = df_f[df_f["risk_level"] == risk_filter]
if min_amount > 0:
    df_f = df_f[df_f["amount"] >= min_amount]
df_f = df_f.head(show_n)

def color_risk(val):
    if val == "HIGH":   return "background-color: #2a1010; color: #f08080"
    elif val == "MEDIUM": return "background-color: #2a1f08; color: #e8c068"
    else:               return "background-color: #0a2e24; color: #7dd4b0"

display_cols = ["id","timestamp","user_id","amount","fraud_probability",
                "risk_level","decision_clean","anomaly","behavior","graph","velocity","device"]
available = [c for c in display_cols if c in df_f.columns]

st.dataframe(
    df_f[available].style.map(color_risk, subset=["risk_level"]),
    use_container_width=True, height=400
)

st.divider()

# ── HIGH RISK CASES WITH LLM REASONING ───────────────────────────────────────
st.subheader("🔴 High Risk Cases — Full Investigation")

high_risk_df = df[df["risk_level"] == "HIGH"].head(10)

if high_risk_df.empty:
    st.info("No HIGH risk cases yet.")
else:
    for _, row in high_risk_df.iterrows():
        decision = clean_decision(row["decision"])
        icon     = "🚫" if decision == "BLOCK" else "⚠️" if decision == "REVIEW" else "✅"

        with st.expander(
            f"{icon} Case #{row['id']} | User: {row['user_id']} | "
            f"Amount: ₹{row['amount']} | Decision: {decision} | {row['timestamp']}"
        ):
            # ── Row 1: Key metrics ──────────────────────────
            c1, c2, c3 = st.columns(3)
            c1.metric("Fraud Probability", row["fraud_probability"])
            c2.metric("Risk Score",        row["risk_score"])
            c3.metric("Decision",          decision)

            st.markdown("---")

            # ── Row 2: Detection signals ────────────────────
            st.markdown("**🔬 Detection Signals:**")
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)

            def signal_badge(label, value, bad_value):
                color = "🔴" if value == bad_value else "🟢"
                return f"{color} **{label}**\n\n`{value}`"

            sc1.markdown(signal_badge("Anomaly",  row["anomaly"],  "ANOMALY"))
            sc2.markdown(signal_badge("Behavior", row["behavior"], "UNUSUAL"))
            sc3.markdown(signal_badge("Graph",    row["graph"],    "FRAUD_NETWORK"))
            sc4.markdown(signal_badge("Velocity", row["velocity"], "SUSPICIOUS"))
            sc5.markdown(signal_badge("Device",   row["device"],   "SUSPICIOUS"))

            st.markdown("---")

            # ── Row 3: SHAP + LLM side by side ─────────────
            shap_col, llm_col = st.columns(2)

            with shap_col:
                st.markdown("**📊 SHAP Explanation** *(why ML model flagged this)*")
                if row.get("shap") and str(row["shap"]) != "nan":
                    st.code(str(row["shap"])[:400], language="text")
                else:
                    st.info("SHAP not available")

            with llm_col:
                st.markdown("**🤖 LLM Reasoning** *(GPT-4o-mini analysis)*")
                reasoning = extract_reasoning(row.get("decision", ""))
                if reasoning and len(reasoning) > 10:
                    st.info(reasoning[:600])
                else:
                    st.info("Full reasoning not available")

st.divider()

# ── MLOPS CONTROLS ────────────────────────────────────────────────────────────
st.subheader("⚙️ MLOps Controls")

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    if st.button("📊 Run Performance Monitor"):
        try:
            import joblib
            from src.preprocessing import load_data, clean_data
            from src.feature_engineering import create_features
            from src.monitor import ModelMonitor

            with st.spinner("Running monitor..."):
                data   = load_data("data/creditcard_augmented.csv")
                data   = clean_data(data)
                data   = create_features(data)
                data   = data.dropna()
                X      = data.drop("Class", axis=1).select_dtypes(include=[float, int])
                y      = data["Class"]
                model  = joblib.load("models/xgboost_model.pkl")
                y_pred = model.predict(X)
                monitor = ModelMonitor()
                report  = monitor.evaluate(y, y_pred)

            st.success("✅ Monitor complete!")
            st.json(report)
        except Exception as e:
            st.error(f"Error: {e}")

with col_m2:
    if st.button("🔍 Check Concept Drift"):
        try:
            from src.preprocessing import load_data, clean_data
            from src.feature_engineering import create_features
            from src.drift_detector import DriftDetector

            with st.spinner("Checking drift..."):
                data  = load_data("data/creditcard_augmented.csv")
                data  = clean_data(data)
                data  = create_features(data)
                data  = data.dropna()
                X     = data.drop("Class", axis=1).select_dtypes(include=[float, int])
                split = int(len(X) * 0.8)
                detector = DriftDetector()
                detector.set_reference(X.iloc[:split])
                status, report = detector.detect_drift(X.iloc[split:])

            if status == "DRIFT_DETECTED":
                st.warning(f"⚠️ {status}")
            else:
                st.success(f"✅ {status}")
            st.json(report)
        except Exception as e:
            st.error(f"Error: {e}")

with col_m3:
    if st.button("🔄 Retrain Model"):
        try:
            from src.retrain_pipeline import RetrainPipeline

            with st.spinner("Retraining... please wait"):
                pipeline = RetrainPipeline()
                deployed, f1_new, f1_old = pipeline.retrain()

            if deployed:
                st.success(f"✅ New model deployed! F1: {f1_new:.4f}")
            else:
                st.warning(f"⚠️ Old model kept. New: {f1_new:.4f} vs Old: {f1_old:.4f}")
        except Exception as e:
            st.error(f"Error: {e}")

st.divider()

if st.button("🔄 Refresh Dashboard"):
    st.cache_data.clear()
    st.rerun()

st.caption("FinShield AI — Student Project | FastAPI + XGBoost + RAG + SHAP + Supabase + Streamlit")