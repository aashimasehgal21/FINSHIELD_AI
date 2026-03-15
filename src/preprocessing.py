import pandas as pd


def load_data(path):

    # load dataset
    df = pd.read_csv(path, low_memory=False)

    # ensure Time column is numeric
    df["Time"] = pd.to_numeric(df["Time"], errors="coerce")

    return df


def clean_data(df):

    print("Original dataset shape:", df.shape)

    # remove duplicates
    df = df.drop_duplicates()

    print("Dataset shape after removing duplicates:", df.shape)

    return df