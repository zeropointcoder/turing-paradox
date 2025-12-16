import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('housing.csv')

# Handle missing values or NaN values
df = df.dropna()

# Encode categorical variable
df = pd.get_dummies(df, columns=['Location'], drop_first=True) # Creates Location_City

# Features and target
X = df.drop('Price', axis=True)
y = df['Price']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit app UI
st.title("Housing Price Predictor")
st.write("Enter house details to predict its price!")

# User inputs
size = st.number_input("Size (sqft)", min_value=500, max_value=5000, value=1500)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
age = st.number_input("House Age (years)", min_value=0, max_value=100, value=10)
location = st.selectbox("Location", ["Suburb", "City"])

# Convert location to model format
location_city = 1 if location == 'City' else 0 # 1=city 0=suburb

# Build input DataFrame
input_data = pd.DataFrame({
    'Size':[size],
    'Bedrooms': [bedrooms],
    'Bathrooms': [bathrooms],
    'Age': [age],
    'Location_City': [location_city]
})

# Reindex to match training columns (prevents feature mismatch errors)
input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

# Prediction button
if st.button("Predict Price"):
    pred_price = model.predict(input_data)[0]
    st.success(f"Predicted house price: £{pred_price:,.2f}")

# Show dataset (optional)
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# Show model evaluation (optional)
if st.checkbox("Show Model Performance"):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**RMSE**: {rmse:,.2f}")
    st.write(f"**R² Score**: {r2:,.2f}")
