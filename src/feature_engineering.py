import pandas as pd


def create_features(df):
    """
    Create fraud detection features
    """

    # High transaction amount flag
    df["high_amount_flag"] = (df["Amount"] > 200).astype(int)

    # Night transaction flag
    df["night_transaction_flag"] = ((df["Time"] % 86400) < 21600).astype(int)

    return df


if __name__ == "__main__":

    df = pd.read_csv("data/creditcard.csv")

    df = create_features(df)

    print("\nNew features added:")
    print(df[["Amount", "high_amount_flag", "night_transaction_flag"]].head())