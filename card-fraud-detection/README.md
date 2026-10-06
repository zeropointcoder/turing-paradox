# Credit Card Fraud Detection

Detect fraudulent credit card transactions using a Random Forest model on synthetic offline data.


## Overview
- Synthetic transaction data is generated with features such as:
    - `amount` – transaction amount
    - `transaction_time` – hour of transaction
    - `customer_age` – age of customer

- Fraud is simulated as a small `percentage` of total transactions.

- Data is standardised using `StandardScaler`:
    $$
    X_{\text{scaled}} = \frac{X - \mu}{\sigma}
    $$

- Model is trained using `Random Forest Classifier`:
    $$
    y_{\text{pred}} = \text{RF}(X_{\text{scaled}})
    $$

- Model evaluation includes:
    - Confusion Matrix
    - Classification Report
    - ROC AUC Score

- NOTE on results:
    The dataset is highly imbalanced `(fraud <5%)`, so the model naturally performs better on legitimate transactions. Lower recall for fraud is expected and reflects real-world credit card fraud detection challenges.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```