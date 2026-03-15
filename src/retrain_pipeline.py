import sys
sys.path.append(".")

import pandas as pd
import joblib
import json
import os
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from dotenv import load_dotenv
from supabase import create_client

from src.preprocessing import load_data, clean_data
from src.feature_engineering import create_features

load_dotenv()

class RetrainPipeline:
    def __init__(self):
        self.model_path    = "models/xgboost_model.pkl"
        self.registry_path = "models/model_registry.json"
        os.makedirs("models", exist_ok=True)

        # connect to Supabase
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        self.supabase = create_client(url, key)

    def load_registry(self):
        if os.path.exists(self.registry_path):
            with open(self.registry_path) as f:
                return json.load(f)
        return []

    def save_registry(self, registry):
        with open(self.registry_path, "w") as f:
            json.dump(registry, f, indent=2)

    def fetch_new_fraud_cases(self):
        """
        Fetch HIGH risk confirmed fraud cases from Supabase.
        These are new patterns the model should learn from.
        """
        print("   Fetching new fraud cases from Supabase...")
        response = self.supabase.table("fraud_cases") \
            .select("*") \
            .eq("risk_level", "HIGH") \
            .execute()

        cases = response.data
        print(f"   Found {len(cases)} HIGH risk cases in Supabase")
        return cases

    def build_new_rows(self, cases):
        """
        Convert Supabase fraud cases into training rows.
        These rows will be added to original training data.
        """
        if not cases:
            return None

        rows = []
        for case in cases:
            row = {
                "Amount": case.get("amount", 0),
                "Time":   0,
                "Class":  1,  # confirmed fraud
                "user_id": case.get("user_id", 0),
            }
            rows.append(row)

        df_new = pd.DataFrame(rows)
        print(f"   Built {len(df_new)} new training rows from Supabase")
        return df_new

    def retrain(self):
        print("=" * 50)
        print("Starting retraining pipeline...")
        print("=" * 50)

        # ── Step 1: Load original data ──────────────────────
        print("\n1. Loading original data...")
        data = load_data("data/creditcard_augmented.csv")
        data = clean_data(data)
        data = create_features(data)
        data = data.dropna()

        # ── Step 2: Fetch new fraud cases from Supabase ──────
        print("\n2. Fetching new data from Supabase...")
        new_cases  = self.fetch_new_fraud_cases()
        new_rows   = self.build_new_rows(new_cases)

        if new_rows is not None and len(new_rows) > 0:
            # fill missing columns with 0
            for col in data.columns:
                if col not in new_rows.columns:
                    new_rows[col] = 0

            # keep only columns that exist in original data
            new_rows = new_rows[[c for c in data.columns if c in new_rows.columns]]

            # merge
            data = pd.concat([data, new_rows], ignore_index=True)
            print(f"   Total training rows after merge: {len(data)}")
        else:
            print("   No new cases — using original data only")

        # ── Step 3: Prepare features ─────────────────────────
        print("\n3. Preparing features...")
        X = data.drop("Class", axis=1).select_dtypes(include=[float, int])
        y = data["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler          = StandardScaler()
        X_train_scaled  = scaler.fit_transform(X_train)
        X_test_scaled   = scaler.transform(X_test)

        smote            = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

        print(f"   Training samples after SMOTE: {len(X_train_res)}")

        # ── Step 4: Train new model ───────────────────────────
        print("\n4. Training new model...")
        new_model = XGBClassifier(
            n_estimators  = 100,
            max_depth     = 6,
            learning_rate = 0.1,
            random_state  = 42,
            eval_metric   = "logloss"
        )
        new_model.fit(X_train_res, y_train_res)

        y_pred_new = new_model.predict(X_test_scaled)
        f1_new     = f1_score(y_test, y_pred_new, zero_division=0)
        print(f"   New model F1: {f1_new:.4f}")

        # ── Step 5: Compare with old model ───────────────────
        print("\n5. Comparing with old model...")
        old_model  = joblib.load(self.model_path)
        y_pred_old = old_model.predict(X_test_scaled)
        f1_old     = f1_score(y_test, y_pred_old, zero_division=0)
        print(f"   Old model F1: {f1_old:.4f}")

        # ── Step 6: Deploy decision ───────────────────────────
        print("\n6. Deploy decision...")
        registry = self.load_registry()
        version  = len(registry) + 1

        if f1_new >= f1_old:
            joblib.dump(new_model, self.model_path)
            joblib.dump(scaler, "models/scaler.pkl")
            deployed = True
            print(f"   ✅ New model DEPLOYED (v{version})")
            print(f"   F1 improved: {f1_old:.4f} → {f1_new:.4f}")
        else:
            deployed = False
            print(f"   ⚠️ Old model KEPT")
            print(f"   New F1 {f1_new:.4f} < Old F1 {f1_old:.4f}")

        # ── Step 7: Save to registry ──────────────────────────
        registry.append({
            "version":        version,
            "timestamp":      datetime.now().isoformat(),
            "f1_new":         round(f1_new, 4),
            "f1_old":         round(f1_old, 4),
            "deployed":       deployed,
            "new_cases_used": len(new_cases)
        })
        self.save_registry(registry)

        print(f"\n7. Registry updated — v{version} saved")
        print("\n" + "=" * 50)
        print("Retraining complete!")
        print("=" * 50)

        return deployed, f1_new, f1_old


if __name__ == "__main__":
    pipeline = RetrainPipeline()
    deployed, f1_new, f1_old = pipeline.retrain()
    print(f"\nResult — Deployed: {deployed} | F1 new: {f1_new} | F1 old: {f1_old}")