import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class DataGenerator:
    def __init__(self, n_samples=200):
        self.n_samples = n_samples

    def generate(self):
        np.random.seed(42)
        engine_size = np.random.uniform(1.0, 6.0, self.n_samples)  # litres
        cylinders = np.random.randint(3, 12, self.n_samples)
        fuel_consumption = engine_size * np.random.uniform(6, 15, self.n_samples)
        co2_emissions = 120 + (engine_size * 30) + (cylinders * 5) + np.random.normal(0, 10, self.n_samples)
        data = pd.DataFrame({
            'Engine Size (L)': engine_size,
            'Cylinders': cylinders,
            'Fuel Consumption (L/100km)': fuel_consumption,
            'CO₂ Emissions (g/km)': co2_emissions
        })
        return data


class CO2Model:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return mse, r2


# Streamlit UI
st.title("Car CO₂ Emissions Predictor")
st.markdown("Predict CO₂ emissions based on engine specifications.")

# Generate dataset
generator = DataGenerator()
df = generator.generate()

st.subheader("Sample Data")
st.dataframe(df.head())

# Train model
features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption (L/100km)']
target = 'CO₂ Emissions (g/km)'

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42
)

model = CO2Model()
model.train(X_train, y_train)

y_pred = model.predict(X_test)
mse, r2 = Evaluator.evaluate(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

st.subheader("Make your own prediction")
engine_input = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
cylinders_input = st.number_input("Cylinders", min_value=3, max_value=12, value=4, step=1)
fuel_input = st.number_input("Fuel Consumption (L/100km)", min_value=3.0, max_value=20.0, value=8.0, step=0.1)

user_data = pd.DataFrame([[engine_input, cylinders_input, fuel_input]], columns=features)
predicted_co2 = model.predict(user_data)[0]

st.success(f"Predicted CO₂ Emissions: {predicted_co2:.2f} g/km")