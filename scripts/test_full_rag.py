import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.retrieval_agent import RetrievalAgent
from src.rag.vector_search import VectorSearch
from src.rag.context_builder import ContextBuilder
from src.rag.decision_agent import DecisionAgent


def run_full_rag_test():

    print("\n========== FULL RAG TEST ==========\n")

    query = "rapid transactions from same account"

    print("Query:")
    print(query)

    # -----------------------
    # Retrieval Agent
    # -----------------------

    print("\n--- Page Index Retrieval ---")

    agent = RetrievalAgent()

    page_rules = agent.retrieve_rules("Card Fraud")

    for r in page_rules[:3]:
        print("-", r)

    # -----------------------
    # Vector Search
    # -----------------------

    print("\n--- Vector DB Retrieval ---")

    vs = VectorSearch()

    vector_results = vs.search(query)

    for r in vector_results:
        print(r)

    # -----------------------
    # Context Builder
    # -----------------------

    print("\n--- Building Context ---")

    context_builder = ContextBuilder()

    context = context_builder.build_context(
        fraud_probability=0.82,
        anomaly_score="High",
        risk_score=67,
        retrieved_rules=page_rules,
        vector_results=vector_results
    )

    print("\nFraud Context:\n")
    print(context)

    # -----------------------
    # LLM Decision
    # -----------------------

    print("\n--- LLM Decision ---")

    agent = DecisionAgent()

    decision = agent.make_decision(context)

    print("\nFinal Decision:\n")
    print(decision)


if __name__ == "__main__":
    run_full_rag_test()