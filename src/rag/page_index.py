from .rule_loader import FraudRuleLoader

class PageIndex:

    def __init__(self):

        loader = FraudRuleLoader()
        self.rules = loader.load_rules()

        self.index = {}

        self.build_index()

    def build_index(self):

        for _, row in self.rules.iterrows():

            fraud_type = row["fraud_type"]

            if fraud_type not in self.index:
                self.index[fraud_type] = []

            self.index[fraud_type].append(row["rule"])

        print("Page index created")

    def get_rules(self, fraud_type):

        return self.index.get(fraud_type, [])


if __name__ == "__main__":

    index = PageIndex()

    rules = index.get_rules("Card Fraud")

    print("Card Fraud Rules:\n")

    for r in rules[:5]:
        print("-", r)