class ContextBuilder:
    def __init__(self):
        pass

    def build_context(
        self,
        fraud_probability,
        anomaly_score,
        behavior_status,
        graph_status,
        velocity_status,
        device_status,
        risk_score,
        risk_level,
        retrieved_rules,
        vector_results,
        shap_explanation=""
    ):
        context = {
            "fraud_probability": fraud_probability,
            "anomaly_score":     anomaly_score,
            "behavior_status":   behavior_status,
            "graph_status":      graph_status,
            "velocity_status":   velocity_status,
            "device_status":     device_status,
            "risk_score":        risk_score,
            "risk_level":        risk_level,
            "retrieved_rules":   retrieved_rules[:3],
            "vector_matches":    vector_results[:3],
            "shap_explanation":  shap_explanation
        }
        return context

    def build_evidence_table(self, context):
        table = f"""
============================================================
         FRAUD INVESTIGATION REPORT
============================================================
ML Fraud Probability  : {context['fraud_probability']}
Anomaly Detection     : {context['anomaly_score']}
Behaviour Profiling   : {context['behavior_status']}
Graph Fraud Detection : {context['graph_status']}
Velocity Check        : {context['velocity_status']}
Device / IP Check     : {context['device_status']}
------------------------------------------------------------
Risk Score            : {context['risk_score']}
Risk Level            : {context['risk_level']}
------------------------------------------------------------
Matched Fraud Rules:
{chr(10).join(f"  - {r}" for r in context['retrieved_rules'])}
------------------------------------------------------------
Similar Fraud Patterns (Vector DB):
{chr(10).join(f"  - {v}" for v in context['vector_matches'])}
------------------------------------------------------------
SHAP Explanation:
{context['shap_explanation'] if context['shap_explanation'] else "  Not available"}
============================================================
"""
        return table


if __name__ == "__main__":
    builder = ContextBuilder()

    context = builder.build_context(
        fraud_probability = 0.82,
        anomaly_score     = "ANOMALY",
        behavior_status   = "UNUSUAL",
        graph_status      = "FRAUD_NETWORK",
        velocity_status   = "SUSPICIOUS",
        device_status     = "SUSPICIOUS",
        risk_score        = 100,
        risk_level        = "HIGH",
        retrieved_rules   = [
            "Multiple transactions within 1 minute",
            "Transaction exceeds historical maximum",
            "Login from unknown device"
        ],
        vector_results    = [
            {"document": "Rapid transactions from same account", "score": 0.91},
            {"document": "High frequency transactions", "score": 0.88}
        ],
        shap_explanation  = "V14 increases fraud risk by 0.32\nAmount increases fraud risk by 0.18"
    )

    table = builder.build_evidence_table(context)
    print(table)