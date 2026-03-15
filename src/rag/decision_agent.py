import os
from dotenv import load_dotenv
from openai import OpenAI

# load environment variables
load_dotenv(".env")


class DecisionAgent:

    def __init__(self):

        api_key = os.getenv("OPENAI_API_KEY")

        if api_key is None:
            raise ValueError("OPENAI_API_KEY not found in .env")

        self.client = OpenAI(api_key=api_key)


    def make_decision(self, context):

        prompt = f"""
You are a fraud detection expert.

Analyze the following fraud evidence and decide whether the transaction should be:

ALLOW
REVIEW
BLOCK

Fraud Evidence:

Fraud Probability: {context['fraud_probability']}
Anomaly Score: {context['anomaly_score']}
Risk Score: {context['risk_score']}

Retrieved Fraud Rules:
{context['retrieved_rules']}

Vector Matches:
{context['vector_matches']}

Provide:
1. Decision
2. Short Reason
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an AI fraud detection analyst."},
                {"role": "user", "content": prompt}
            ]
        )

        decision = response.choices[0].message.content

        return decision


if __name__ == "__main__":

    agent = DecisionAgent()

    sample_context = {
        "fraud_probability": 0.82,
        "anomaly_score": "High",
        "risk_score": 67,
        "retrieved_rules": [
            "Multiple transactions within 1 minute",
            "Transaction exceeds historical maximum"
        ],
        "vector_matches": [
            {"document": "Rapid transactions from same account", "score": 0.91},
            {"document": "High frequency transactions", "score": 0.88}
        ]
    }

    result = agent.make_decision(sample_context)

    print("\nLLM Decision:\n")
    print(result)