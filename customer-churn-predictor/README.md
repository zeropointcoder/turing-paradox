# Customer Churn Predictor

A command-line machine learning application that predicts customer churn using structured behavioural data.


## Overview
- Synthetic customer data is generated offline using `multivariate feature distributions`

- Each record represents a customer with behavioural indicators

- `Logistic Regression` is trained to estimate churn probability

- Binary classification is performed using the `sigmoid` function:
    `σ(z) = 1 / (1 + e⁻ᶻ)`

- Model performance is evaluated using:
  - Accuracy
  - Precision
  - Recall
  - `F1`-score
 

## Run
```bash
pip install -r requirements.txt
```

```bash
python3 customer_churn_predictor.py
```