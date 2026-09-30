# AI-Driven Portfolio Tracker

An AI-driven system that tracks, analyses and forecasts portfolio performance using fully local machine-learning models and an interactive dashboard.


## Overview
- Synthetic market prices are generated using stochastic return processes

- Daily returns are calculated as:
    `rₜ = (Pₜ − Pₜ₋₁) / Pₜ₋₁`

- A rolling window of historical returns is used as model input features

- A `linear regression model` learns the relationship between past returns and future returns

- Predictions are evaluated using Root Mean Squared Error `(RMSE)`:  
    `RMSE = √(1/n Σ(y − ŷ)²)`


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```