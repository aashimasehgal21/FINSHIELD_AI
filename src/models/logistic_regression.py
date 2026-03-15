from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from src.train_model import get_training_data


def train_logistic():

    X_train, X_test, y_train, y_test = get_training_data()

    model = LogisticRegression(max_iter=2000)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nLogistic Regression Performance:\n")

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_logistic()