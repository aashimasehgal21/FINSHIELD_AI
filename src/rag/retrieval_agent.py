from .page_index import PageIndex


class RetrievalAgent:

    def __init__(self):

        self.index = PageIndex()

    def retrieve_rules(self, fraud_type):

        rules = self.index.get_rules(fraud_type)

        return rules


if __name__ == "__main__":

    agent = RetrievalAgent()

    rules = agent.retrieve_rules("Card Fraud")

    print("\nRetrieved Rules:\n")

    for r in rules[:5]:
        print("-", r)