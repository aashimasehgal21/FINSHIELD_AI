import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.train_model import get_training_data


def evaluate_model(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return precision, recall


def compare_models():

    X_train, X_test, y_train, y_test = get_training_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    results = []

    for name, model in models.items():

        precision, recall = evaluate_model(
            model, X_train, X_test, y_train, y_test
        )

        results.append([name, precision, recall])

    df = pd.DataFrame(results, columns=["Model", "Precision", "Recall"])

    print("\nModel Comparison:\n")
    print(df)

    # BAR GRAPH
    df.set_index("Model")[["Precision", "Recall"]].plot(
        kind="bar",
        figsize=(8,5)
    )

    plt.title("Fraud Detection Model Comparison")
    plt.ylabel("Score")
    plt.xlabel("Model")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    compare_models()