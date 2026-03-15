import sys
sys.path.append(".")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

from src.preprocessing import load_data, clean_data
from src.feature_engineering import create_features


def get_training_data():

    df = load_data("data/creditcard_augmented.csv")

    df = clean_data(df)

    df = create_features(df)

    # remove rows with missing values
    df = df.dropna()

    X = df.drop("Class", axis=1)
    # drop non-numeric columns — model only works with numbers
    X = X.select_dtypes(include=[float, int])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE:")
    print(pd.Series(y_train).value_counts())

    return X_train, X_test, y_train, y_test


def train_model():

    print("Preparing data...")

    X_train, X_test, y_train, y_test = get_training_data()

    print("Training XGBoost model...")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("Training completed")

    # save model
    joblib.dump(model, "models/xgboost_model.pkl")

    print("Model saved to models/xgboost_model.pkl")


if __name__ == "__main__":
    train_model()