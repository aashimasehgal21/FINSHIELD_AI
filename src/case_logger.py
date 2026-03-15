import os
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

class CaseLogger:
    def __init__(self):
        self.table = "fraud_cases"
        print("CaseLogger: Supabase connected")

    def log(self, user_id, amount, result):
        """
        Save one fraud case to Supabase.
        """
        record = {
            "timestamp":        datetime.now().isoformat(),
            "user_id":          int(user_id),
            "amount":           float(amount),
            "fraud_probability": float(result.get("fraud_probability", 0)),
            "anomaly":          str(result.get("anomaly", "")),
            "behavior":         str(result.get("behavior", "")),
            "graph":            str(result.get("graph", "")),
            "velocity":         str(result.get("velocity", "")),
            "device":           str(result.get("device", "")),
            "risk_score":       float(result.get("risk_score", 0)),
            "risk_level":       str(result.get("risk_level", "")),
            "decision":         str(result.get("decision", "")),
            "shap":             str(result.get("shap", ""))
        }

        response = supabase.table(self.table).insert(record).execute()
        print(f"Logged case — User: {user_id}, Amount: {amount}, Risk: {result.get('risk_level')}")
        return response

    def get_all(self):
        """
        Get all fraud cases from Supabase.
        """
        response = supabase.table(self.table).select("*").order("id", desc=True).execute()
        return response.data

    def get_high_risk(self):
        """
        Get only HIGH risk cases.
        """
        response = supabase.table(self.table).select("*").eq("risk_level", "HIGH").execute()
        return response.data

    def get_stats(self):
        """
        Get summary stats for dashboard.
        """
        all_cases  = self.get_all()
        total      = len(all_cases)
        high_risk  = len([c for c in all_cases if c["risk_level"] == "HIGH"])
        medium     = len([c for c in all_cases if c["risk_level"] == "MEDIUM"])
        low        = len([c for c in all_cases if c["risk_level"] == "LOW"])
        blocked    = len([c for c in all_cases if "BLOCK" in str(c["decision"])])

        return {
            "total":     total,
            "high_risk": high_risk,
            "medium":    medium,
            "low":       low,
            "blocked":   blocked
        }


if __name__ == "__main__":
    logger = CaseLogger()

    # test log
    logger.log(
        user_id = 103,
        amount  = 5000.0,
        result  = {
            "fraud_probability": 0.82,
            "anomaly":           "ANOMALY",
            "behavior":          "UNUSUAL",
            "graph":             "FRAUD_NETWORK",
            "velocity":          "SUSPICIOUS",
            "device":            "SUSPICIOUS",
            "risk_score":        100,
            "risk_level":        "HIGH",
            "decision":          "BLOCK",
            "shap":              "Amount increases risk by 0.32"
        }
    )

    print("\nAll cases:")
    cases = logger.get_all()
    for c in cases[:3]:
        print(c)

    print("\nStats:")
    print(logger.get_stats())