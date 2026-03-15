from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from src.train_model import get_training_data


def train_xgboost():

    X_train, X_test, y_train, y_test = get_training_data()

    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nXGBoost Performance:\n")

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_xgboost()

import joblib
from xgboost import XGBClassifier

def train_xgboost(X_train, y_train):

    model = XGBClassifier()

    model.fit(X_train, y_train)

    # save model
    joblib.dump(model, "models/xgboost_model.pkl")

    return model   