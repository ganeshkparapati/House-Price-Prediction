import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib


def train_model():
    df = pd.read_csv("data/housing.csv")

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, "data/house_price_model.pkl")
    print("Model trained and saved successfully!")


if __name__ == "__main__":
    train_model()