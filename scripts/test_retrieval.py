import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.rag.rule_loader import FraudRuleLoader
from src.rag.page_index import PageIndex
from src.rag.retrieval_agent import RetrievalAgent
from src.rag.vector_search import VectorSearch


def run_retrieval_test():

    print("\n========== RETRIEVAL TEST ==========\n")

    # ------------------------
    # Load Fraud Rules
    # ------------------------

    print("--- Loading Fraud Rules ---")

    loader = FraudRuleLoader()
    rules = loader.load_rules()

    print("Total Rules:", len(rules))

    # ------------------------
    # Page Index Test
    # ------------------------

    print("\n Page Index Retrieval ")

    index = PageIndex()

    agent = RetrievalAgent()

    card_rules = agent.retrieve_rules("Card Fraud")

    print("\nSample Retrieved Rules:")

    for r in card_rules[:3]:
        print("-", r)

    # ------------------------
    # Vector DB Retrieval Test
    # ------------------------

    print("\n Vector Search Retrieval ")

    vs = VectorSearch()

    query = "rapid transactions from same account"

    print("\nQuery:", query)

    results = vs.search(query)

    print("\nVector Search Results:\n")

    for r in results:
        print(r)


if __name__ == "__main__":
    run_retrieval_test()