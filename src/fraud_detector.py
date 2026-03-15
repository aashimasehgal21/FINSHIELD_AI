import joblib
from src.train_model import get_training_data


# load trained model
model = joblib.load("models/xgboost_model.pkl")


def predict_fraud(sample):

    probability = model.predict_proba(sample)[0][1]

    return probability


if __name__ == "__main__":

    print("Preparing sample transaction...")

    # reuse training pipeline so features match
    X_train, X_test, y_train, y_test = get_training_data()

    sample_transaction = X_test[0:1]

    probability = predict_fraud(sample_transaction)

    print("\nFraud Probability:", probability)

    if probability > 0.8:
        print("Risk Level: HIGH")
    elif probability > 0.4:
        print("Risk Level: MEDIUM")
    else:
        print("Risk Level: LOW")