import pandas as pd


class FraudRuleLoader:

    def __init__(self, path="data/fraud_rules.csv"):
        self.path = path
        self.rules = None

    def load_rules(self):

        self.rules = pd.read_csv(self.path)

        print("Fraud rules loaded successfully")
        print("Total rules:", len(self.rules))

        return self.rules


if __name__ == "__main__":

    loader = FraudRuleLoader()

    rules = loader.load_rules()

    print("\nSample rules:\n")

    print(rules.head())