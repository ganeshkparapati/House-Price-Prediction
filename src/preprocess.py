import pandas as pd


def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Remove missing values
    df = df.dropna()

    return df


if __name__ == "__main__":
    df = preprocess_data("data/housing.csv")
    print(df.head())