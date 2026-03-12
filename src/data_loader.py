import pandas as pd


def load_dataset(path):

    print("Loading dataset...")

    df = pd.read_csv(path)

    print("Dataset shape:", df.shape)

    df = df.dropna(subset=["listed_in", "description"])

    print("Dataset after dropping missing values:", df.shape)

    return df