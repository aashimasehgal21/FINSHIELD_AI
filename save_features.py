import sys
sys.path.append(".")

import joblib
import json
import pandas as pd
from src.preprocessing import load_data, clean_data
from src.feature_engineering import create_features

# load data
data = load_data("data/creditcard_augmented.csv")
data = clean_data(data)
data = create_features(data)
data = data.dropna()

X = data.drop("Class", axis=1)
X = X.select_dtypes(include=[float, int])

feature_names = list(X.columns)

# save feature names
with open("models/feature_names.json", "w") as f:
    json.dump(feature_names, f)

print(f"Saved {len(feature_names)} feature names:")
print(feature_names)