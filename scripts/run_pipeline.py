import pandas as pd
import sys
import os
import joblib

# add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.risk_engine import RiskEngine
from src.anomaly_detector import AnomalyDetector
from src.behavior_profiler import BehaviorProfiler


print("Step 1: Loading dataset")

data = pd.read_csv("data/creditcard.csv", nrows=2000)

# -------------------------
# Feature preparation
# -------------------------

X = data.drop("Class", axis=1)

# convert everything to numeric first
X = X.apply(pd.to_numeric, errors="coerce")

# fill missing values
X = X.fillna(0)

# -------------------------
# Feature engineering
# -------------------------

X["high_amount_flag"] = (X["Amount"] > 2000).astype(int)

# ensure Time is numeric
X["night_transaction_flag"] = (X["Time"] % 86400 < 21600).astype(int)


print("Step 2: Loading fraud model")

model = joblib.load("models/xgboost_model.pkl")


print("Step 3: Fraud prediction")

sample = X.iloc[[0]]

fraud_prob = model.predict_proba(sample)[0][1]

print("Fraud Probability:", fraud_prob)


# -------------------------
# Anomaly Detection
# -------------------------

print("Step 4: Anomaly detection")

anomaly = AnomalyDetector()

anomaly.train(X.sample(1000))

anomaly_result = anomaly.detect(sample)

print("Anomaly Status:", anomaly_result)


# -------------------------
# Behavior Profiling
# -------------------------

print("Step 5: Behavior profiling")

profiler = BehaviorProfiler()

profiler.train(data)

behavior_result = profiler.check_behavior(sample)

print("Behavior Status:", behavior_result)
print("Step 6: Risk evaluation")

risk_engine = RiskEngine()

risk_score, risk_level = risk_engine.calculate_risk(
    fraud_prob,
    anomaly_result,
    behavior_result
)

print("Risk Score:", risk_score)
print("Risk Level:", risk_level)

print("\nFraud Detection Pipeline Completed")