import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=0.01,
            random_state=42
        )

    def train(self, data):
        """
        Train anomaly detection model on transaction data
        """
        self.model.fit(data)

    def detect(self, transaction):
        """
        Detect if transaction is anomaly
        """
        prediction = self.model.predict(transaction)

        if prediction[0] == -1:
            return "ANOMALY"
        else:
            return "NORMAL"