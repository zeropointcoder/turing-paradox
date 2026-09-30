import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class SyntheticMarketData:
    def __init__(self, assets=3, days=260, seed=42):
        self.assets = assets
        self.days = days
        self.seed = seed

    def generate(self):
        np.random.seed(self.seed)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=self.days, freq="B")
        data = {}

        for i in range(self.assets):
            returns = np.random.normal(0.0005, 0.01, self.days)
            prices = 100 * np.cumprod(1 + returns)
            data[f"Asset_{i+1}"] = prices

        return pd.DataFrame(data, index=dates)


class PortfolioModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class PortfolioEvaluator:
    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))


class PortfolioDashboard:
    def __init__(self):
        st.set_page_config(page_title="AI Portfolio Tracker", layout="wide")
        st.title("AI-Driven Portfolio Tracker")

    def run(self):
        data_generator = SyntheticMarketData()
        prices = data_generator.generate()

        st.subheader("Asset Price History")
        st.line_chart(prices)

        asset = st.selectbox("Select target asset", prices.columns)
        window = st.slider("Lookback window (days)", 5, 30, 10)

        returns = prices.pct_change().dropna()
        X, y = [], []

        for i in range(window, len(returns)):
            X.append(returns.iloc[i - window:i].values.flatten())
            y.append(returns[asset].iloc[i])

        X = np.array(X)
        y = np.array(y)

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = PortfolioModel()
        model.train(X_train, y_train)

        predictions = model.predict(X_test)
        score = PortfolioEvaluator.rmse(y_test, predictions)

        st.subheader("Prediction Performance")
        st.metric("RMSE", f"{score:.6f}")

        comparison = pd.DataFrame({
            "Actual": y_test,
            "Predicted": predictions 
        })

        st.line_chart(comparison)


if __name__ == "__main__":
    PortfolioDashboard().run()