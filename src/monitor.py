import json
import os
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score

class ModelMonitor:
    def __init__(self, threshold_f1=0.85, log_path="logs/performance_log.json"):
        self.threshold_f1 = threshold_f1
        self.log_path     = log_path
        os.makedirs("logs", exist_ok=True)

    def evaluate(self, y_true, y_pred):
        f1   = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)

        needs_retraining = f1 < self.threshold_f1

        report = {
            "timestamp":         datetime.now().isoformat(),
            "f1":                round(f1, 4),
            "precision":         round(prec, 4),
            "recall":            round(rec, 4),
            "needs_retraining":  needs_retraining
        }

        # save to log file
        logs = []
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                logs = json.load(f)
        logs.append(report)
        with open(self.log_path, "w") as f:
            json.dump(logs, f, indent=2)

        print(f"F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
        print(f"Needs retraining: {needs_retraining}")
        return report

    def get_latest(self):
        if not os.path.exists(self.log_path):
            return None
        with open(self.log_path) as f:
            logs = json.load(f)
        return logs[-1] if logs else None


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    import pandas as pd
    import joblib
    from src.preprocessing import load_data, clean_data
    from src.feature_engineering import create_features

    print("Testing ModelMonitor...")

    data = load_data("data/creditcard_augmented.csv")
    data = clean_data(data)
    data = create_features(data)
    data = data.dropna()

    X = data.drop("Class", axis=1).select_dtypes(include=[float, int])
    y = data["Class"]

    model  = joblib.load("models/xgboost_model.pkl")
    y_pred = model.predict(X)

    monitor = ModelMonitor()
    report  = monitor.evaluate(y, y_pred)
    print("\nReport saved to logs/performance_log.json")