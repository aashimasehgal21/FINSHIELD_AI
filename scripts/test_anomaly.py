import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.anomaly_detector import AnomalyDetector

print("Step 1: Loading dataset")

data = pd.read_csv("data/creditcard.csv", nrows=2000)

X = data.drop("Class", axis=1)

# clean dataset
X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

print("Step 2: Training anomaly detector")

detector = AnomalyDetector()
detector.train(X)

print("Step 3: Testing transaction")

sample = X.iloc[[0]]

result = detector.detect(sample)

print("Transaction Status:", result)