# Time Series Forecasting: ARIMA

Predict future values of a synthetic time series using ARIMA models.


## Overview
- **Data Generation**: Linear trend + sine wave + Gaussian noise

- **ARIMA**: Auto-Regressive Integrated Moving Average
    - Formula: `y_t = c + φ1*y_{t-1} + ... + θ1*ε_{t-1} + ε_t`

- **Forecasting**: User selects number of future steps

- **Visualisation**: Streamlit charts for original and forecasted values


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```