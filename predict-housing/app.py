import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class HousingDataset:
    def __init__(self, samples: int = 500, seed: int = 42):
        np.random.seed(seed)
        self.samples = samples

    def generate(self) -> pd.DataFrame:
        size = np.random.randint(40, 200, self.samples)
        bedrooms = np.random.randint(1, 6, self.samples)
        age = np.random.randint(0, 80, self.samples)

        price = (
            size * 3000 +
            bedrooms * 25000 -
            age * 1200 +
            np.random.normal(0, 20000, self.samples)
        )

        return pd.DataFrame({
            "size_sq_m": size,
            "bedrooms": bedrooms,
            "age_years": age,
            "price_gbp": price
        })


class HousingModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class ModelEvaluator:
    @staticmethod
    def evaluate(y_true, y_pred) -> dict:
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        return {
            "rmse": rmse,
            "r2": r2
        }


class HousingApp:
    def __init__(self):
        st.set_page_config(page_title="Housing Price Predictor", layout="centered")
        st.title("Housing Price Predictor")
        st.write("Predict UK house prices using linear regression.")

        self.dataset = HousingDataset()
        self.model = HousingModel()

    def run(self):
        data = self.dataset.generate()

        X = data[["size_sq_m", "bedrooms", "age_years"]]
        y = data["price_gbp"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.train(X_train, y_train)
        predictions = self.model.predict(X_test)

        metrics = ModelEvaluator.evaluate(y_test, predictions)

        st.subheader("Model Performance")
        st.metric("RMSE (£)", f"{metrics['rmse']:,.0f}")
        st.metric("R² Score", f"{metrics['r2']:.2f}")

        st.subheader("Predict a Property Price")

        size = st.slider("Property size (square metres)", 40, 200, 90)
        bedrooms = st.slider("Number of bedrooms", 1, 5, 3)
        age = st.slider("Property age (years)", 0, 80, 10)

        input_df = pd.DataFrame([[size, bedrooms, age]],
                                columns=X.columns)

        prediction = self.model.predict(input_df)[0]

        st.success(f"Estimated price: £{prediction:,.0f}")


if __name__ == "__main__":
    HousingApp().run()
