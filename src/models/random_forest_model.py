from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from src.train_model import get_training_data


def train_random_forest():

    X_train, X_test, y_train, y_test = get_training_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nRandom Forest Performance:\n")

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_random_forest()