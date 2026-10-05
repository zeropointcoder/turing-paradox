import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Data Generation class
class DataGenerator:
    def __init__(self, n_samples=200):
        self.n_samples = n_samples

    def generate(self):
        np.random.seed(42)
        base_price = np.random.uniform(5, 100, self.n_samples)
        demand_factor = np.random.uniform(50, 200, self.n_samples)
        noise = np.random.normal(0, 10, self.n_samples)
        sales = demand_factor - 1.5 * base_price + noise
        sales = np.clip(sales, 0, None)
        df = pd.DataFrame({'Base_Price': base_price, 'Sales': sales})
        df['Revenue'] = df['Base_Price'] * df['Sales']
        return df


# Price Prediction Model class
class PriceModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        # Ensure X is a DataFrame with proper column names
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=['Base_Price'])
        return self.model.predict(X)


# Evaluation class
class Evaluator:
    @staticmethod
    def mse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)


# Price Optimiser class
class Optimiser:
    def __init__(self, model):
        self.model = model

    def optimal_price(self, min_price=5, max_price=100):
        prices = np.linspace(min_price, max_price, 100)
        df_prices = pd.DataFrame(prices, columns=['Base_Price'])
        predicted_sales = self.model.predict(df_prices)
        revenue = prices * predicted_sales
        idx = np.argmax(revenue)
        return prices[idx], revenue[idx], prices, revenue


# Streamlit Interface
st.title("Product Price Optimiser")

st.sidebar.header("Settings")
num_samples = st.sidebar.slider("Number of generated data points", 50, 500, 200)

# Generate data
data_gen = DataGenerator(num_samples)
data = data_gen.generate()
st.subheader("Generated Data")
st.dataframe(data.head())

# Train model
model = PriceModel()
model.train(data[['Base_Price']], data['Sales'])

# Evaluate
y_pred = model.predict(data[['Base_Price']])
mse_val = Evaluator.mse(data['Sales'], y_pred)
st.write(f"Model MSE: {mse_val:.2f}")

# Optimise price
optimiser = Optimiser(model)
opt_price, opt_revenue, prices, revenues = optimiser.optimal_price()
st.success(f"Optimal Price: £{opt_price:.2f} | Expected Revenue: £{opt_revenue:.2f}")

# Revenue curve
st.subheader("Revenue Curve")
fig, ax = plt.subplots()
ax.plot(prices, revenues, color='green')
ax.axvline(opt_price, color='red', linestyle='--', label='Optimal Price')
ax.set_xlabel("Price (£)")
ax.set_ylabel("Revenue (£)")
ax.legend()
st.pyplot(fig)