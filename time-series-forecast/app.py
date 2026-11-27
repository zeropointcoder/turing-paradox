import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# DATA GENERATION
class TimeSeriesData:
    def __init__(self, n_points=200):
        self.n_points = n_points

    def generate(self):
        np.random.seed(42)
        t = np.arange(self.n_points)
        data = 10 + 0.5*t + np.sin(0.2*t) + np.random.normal(scale=2, size=self.n_points)
        df = pd.DataFrame({'Time': t, 'Value': data})
        return df


# ARIMA MODEL
class ARIMAModel:
    def __init__(self, order=(5,1,0)):
        self.order = order
        self.model = None

    def fit(self, series):
        self.model = ARIMA(series, order=self.order).fit()
        return self.model

    def predict(self, steps=10):
        return self.model.forecast(steps=steps)


# STREAMLIT APP
st.set_page_config(page_title="Time Series Forecasting", layout="wide")
st.title("Time Series Forecasting with ARIMA")

# Generate synthetic data
ts_data = TimeSeriesData()
df = ts_data.generate()

st.subheader("Generated Time Series Data")
st.line_chart(df.set_index('Time'))

# ARIMA Forecast
st.subheader("ARIMA Forecast")
arima_model = ARIMAModel(order=(5,1,0))
arima_model.fit(df['Value'])
arima_steps = st.slider("Forecast steps", min_value=1, max_value=50, value=10)
arima_pred = arima_model.predict(steps=arima_steps)
arima_df = pd.DataFrame({'Forecast': arima_pred}, index=range(len(df), len(df)+arima_steps))
st.line_chart(arima_df)
