import sys
sys.path.append(".")

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import json

from src.anomaly_detector import AnomalyDetector
from src.behavior_profiler import BehaviorProfiler
from src.graph_detector import GraphDetector
from src.velocity_checker import VelocityChecker
from src.device_ip_checker import DeviceIPChecker
from src.risk_engine import RiskEngine
from src.feature_engineering import create_features
from src.case_logger import CaseLogger
from src.rag.retrieval_agent import RetrievalAgent
from src.rag.context_builder import ContextBuilder
from src.rag.decision_agent import DecisionAgent

app = FastAPI(title="FinShield AI", description="Real-Time Fraud Detection API")

print("Loading data and training models...")
data = pd.read_csv("data/creditcard_augmented.csv", nrows=5000)

# add engineered features for anomaly detector
data_fe = create_features(data.copy())
data_fe = data_fe.dropna()

# train anomaly detector WITH engineered features
anomaly = AnomalyDetector()
anomaly_features = data_fe.drop("Class", axis=1).select_dtypes(include=[float, int])
anomaly.train(anomaly_features)

profiler = BehaviorProfiler()
profiler.train(data)

graph = GraphDetector()
graph.build_graph(data)

velocity = VelocityChecker()
velocity.train(data)

device_ip = DeviceIPChecker()
device_ip.train(data)

risk_engine = RiskEngine()
risk_engine.load_model()

model = joblib.load("models/xgboost_model.pkl")

# load feature names
with open("models/feature_names.json", "r") as f:
    model_features = json.load(f)

# case logger — Supabase
case_logger = CaseLogger()

print(f"Loaded {len(model_features)} feature names")
print("All models loaded! Server ready.")

class Transaction(BaseModel):
    Amount: float
    Time: Optional[float] = 0.0
    user_id: Optional[int] = 1
    merchant_id: Optional[int] = 1
    device_id: Optional[str] = "DEV_1"
    ip_address: Optional[str] = "192.168.1.1"
    country: Optional[str] = "IN"
    timestamp: Optional[str] = "2024-01-01 00:00:00"

    class Config:
        extra = "allow"

@app.post("/predict")
def predict(transaction: Transaction):
    txn_dict = dict(transaction)
    df_txn   = pd.DataFrame([txn_dict])

    # get numeric columns
    X = df_txn.select_dtypes(include=[float, int])

    # add missing model features with 0, keep exact order
    for col in model_features:
        if col not in X.columns:
            X[col] = 0
    X = X[model_features]

    # fraud probability
    fraud_prob = float(model.predict_proba(X)[0][1])

    # anomaly — needs engineered features
    X_anomaly = X.copy()
    if "Amount" in X_anomaly.columns:
        X_anomaly["high_amount_flag"] = (X_anomaly["Amount"] > 2000).astype(int)
    else:
        X_anomaly["high_amount_flag"] = 0
    if "Time" in X_anomaly.columns:
        X_anomaly["night_transaction_flag"] = (X_anomaly["Time"] % 86400 < 21600).astype(int)
    else:
        X_anomaly["night_transaction_flag"] = 0

    anomaly_result  = anomaly.detect(X_anomaly)
    behavior_result = profiler.check_behavior(df_txn)
    graph_result    = graph.detect(df_txn)
    velocity_result = velocity.check(df_txn)
    device_result   = device_ip.check(df_txn)

    # risk score
    risk_score, risk_level = risk_engine.calculate_risk(
        fraud_prob      = fraud_prob,
        anomaly_status  = anomaly_result,
        behavior_status = behavior_result,
        graph_status    = graph_result,
        velocity_status = velocity_result,
        device_status   = device_result
    )

    # SHAP
    shap_explanation = risk_engine.explain(X)

    # RAG
    retrieval_agent = RetrievalAgent()
    rules = retrieval_agent.retrieve_rules("Card Fraud")

    try:
        from src.rag.vector_search import VectorSearch
        vs = VectorSearch()
        vector_results = vs.search("rapid transactions")
    except:
        vector_results = []

    # context
    builder = ContextBuilder()
    context = builder.build_context(
        fraud_probability = round(fraud_prob, 4),
        anomaly_score     = anomaly_result,
        behavior_status   = behavior_result,
        graph_status      = graph_result,
        velocity_status   = velocity_result,
        device_status     = device_result,
        risk_score        = risk_score,
        risk_level        = risk_level,
        retrieved_rules   = rules,
        vector_results    = vector_results,
        shap_explanation  = shap_explanation
    )

    # LLM decision
    try:
        decision_agent = DecisionAgent()
        decision = decision_agent.make_decision(context)
    except:
        decision = "REVIEW"

    # log to Supabase
    try:
        case_logger.log(
            user_id = txn_dict.get("user_id", 0),
            amount  = txn_dict.get("Amount", 0),
            result  = {
                "fraud_probability": round(fraud_prob, 4),
                "anomaly":           anomaly_result,
                "behavior":          behavior_result,
                "graph":             graph_result,
                "velocity":          velocity_result,
                "device":            device_result,
                "risk_score":        risk_score,
                "risk_level":        risk_level,
                "decision":          decision,
                "shap":              shap_explanation
            }
        )
    except Exception as e:
        print(f"Logging failed: {e}")

    return {
        "fraud_probability": round(fraud_prob, 4),
        "anomaly":           anomaly_result,
        "behavior":          behavior_result,
        "graph":             graph_result,
        "velocity":          velocity_result,
        "device":            device_result,
        "risk_score":        risk_score,
        "risk_level":        risk_level,
        "decision":          decision,
        "shap":              shap_explanation
    }

@app.get("/")
def home():
    return {"message": "FinShield AI is running!"}