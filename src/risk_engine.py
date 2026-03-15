import sys
sys.path.append(".")

import numpy as np
import joblib
import shap

class RiskEngine:
    def __init__(self):
        self.model = None
        self.explainer = None

    def load_model(self):
        try:
            self.model = joblib.load("models/xgboost_model.pkl")
            self.explainer = shap.TreeExplainer(self.model)
            print("RiskEngine: model loaded for SHAP")
        except Exception as e:
            print(f"RiskEngine: could not load model — {e}")
            self.model = None
            self.explainer = None

    def calculate_risk(self, fraud_prob, anomaly_status, behavior_status,
                       graph_status="CLEAN", velocity_status="NORMAL",
                       device_status="CLEAN"):
        # base score from ML model
        risk_score = fraud_prob * 100

        if anomaly_status == "ANOMALY":
            risk_score += 15

        if behavior_status == "UNUSUAL":
            risk_score += 10

        if graph_status == "FRAUD_NETWORK":
            risk_score += 20

        if velocity_status == "SUSPICIOUS":
            risk_score += 15

        if device_status == "SUSPICIOUS":
            risk_score += 10

        # cap at 100
        risk_score = min(risk_score, 100)

        if risk_score >= 75:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return round(risk_score, 2), risk_level

    def explain(self, sample):
        if self.explainer is None:
            return "SHAP explainer not available"

        try:
            # get model's expected feature names
            model_features = self.model.get_booster().feature_names

            # keep only columns model was trained on
            if model_features:
                available = [f for f in model_features if f in sample.columns]
                sample = sample[available]

            shap_values = self.explainer.shap_values(sample)

            # for binary classification shap_values is a list — index 1 = fraud
            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                values = shap_values[0]

            feature_names = sample.columns.tolist()
            shap_pairs = list(zip(feature_names, values))

            # sort by absolute value — biggest impact first
            shap_pairs.sort(key=lambda x: abs(x[1]), reverse=True)

            top5 = shap_pairs[:5]

            explanation = "Top reasons for this prediction:\n"
            for feat, val in top5:
                direction = "increases" if val > 0 else "decreases"
                explanation += f"  {feat}: {direction} fraud risk by {abs(val):.4f}\n"

            return explanation

        except Exception as e:
            return f"SHAP explanation failed: {e}"


if __name__ == "__main__":
    import pandas as pd
    from src.preprocessing import load_data, clean_data
    from src.feature_engineering import create_features

    print("Testing RiskEngine with SHAP...")

    # test risk calculation
    engine = RiskEngine()

    score, level = engine.calculate_risk(
        fraud_prob      = 0.82,
        anomaly_status  = "ANOMALY",
        behavior_status = "UNUSUAL",
        graph_status    = "FRAUD_NETWORK",
        velocity_status = "SUSPICIOUS",
        device_status   = "SUSPICIOUS"
    )
    print(f"Risk Score : {score}")
    print(f"Risk Level : {level}")

    # test SHAP
    engine.load_model()

    if engine.model is not None:
        data = load_data("data/creditcard_augmented.csv")
        data = clean_data(data)
        data = create_features(data)
        data = data.dropna()

        X = data.drop("Class", axis=1)
        X = X.select_dtypes(include=[float, int])

        sample = X.iloc[[0]]
        explanation = engine.explain(sample)
        print("\n" + explanation)
    else:
        print("Skipping SHAP — model not loaded")