import sys
sys.path.append(".")

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from scipy import stats

class DriftDetector:
    def __init__(self, log_path="logs/drift_log.json"):
        self.reference_stats = {}
        self.log_path        = log_path
        os.makedirs("logs", exist_ok=True)

    def set_reference(self, X_train):
        """
        Save the training data distribution as reference.
        Run this once after training.
        """
        for col in X_train.columns:
            self.reference_stats[col] = {
                "mean": float(X_train[col].mean()),
                "std":  float(X_train[col].std()),
                "data": X_train[col].tolist()
            }
        print(f"DriftDetector: reference set for {len(self.reference_stats)} features")

    def detect_drift(self, X_new):
        """
        Compare new data distribution to reference.
        Returns: DRIFT_DETECTED or NO_DRIFT
        """
        drifted = []

        for col in X_new.columns:
            if col not in self.reference_stats:
                continue

            ref_data = self.reference_stats[col]["data"]
            new_data = X_new[col].tolist()

            # KS test — compares two distributions
            stat, p_value = stats.ks_2samp(ref_data[:1000], new_data[:1000])

            # if p_value < 0.05 — distributions are significantly different
            if p_value < 0.05:
                drifted.append({
                    "feature": col,
                    "p_value": round(p_value, 6),
                    "ks_stat": round(stat, 6)
                })

        status = "DRIFT_DETECTED" if len(drifted) > 0 else "NO_DRIFT"

        report = {
            "timestamp":      datetime.now().isoformat(),
            "status":         status,
            "drifted_count":  len(drifted),
            "drifted_features": drifted[:5]
        }

        # save log
        logs = []
        if os.path.exists(self.log_path):
            with open(self.log_path) as f:
                logs = json.load(f)
        logs.append(report)
        with open(self.log_path, "w") as f:
            json.dump(logs, f, indent=2)

        print(f"Drift status: {status}")
        if drifted:
            print(f"Drifted features: {[d['feature'] for d in drifted[:3]]}")

        return status, report


if __name__ == "__main__":
    from src.preprocessing import load_data, clean_data
    from src.feature_engineering import create_features

    print("Testing DriftDetector...")

    data = load_data("data/creditcard_augmented.csv")
    data = clean_data(data)
    data = create_features(data)
    data = data.dropna()

    X = data.drop("Class", axis=1).select_dtypes(include=[float, int])

    # use first 80% as reference, last 20% as new data
    split = int(len(X) * 0.8)
    X_train = X.iloc[:split]
    X_new   = X.iloc[split:]

    detector = DriftDetector()
    detector.set_reference(X_train)
    status, report = detector.detect_drift(X_new)
    print(f"\nResult: {status}")