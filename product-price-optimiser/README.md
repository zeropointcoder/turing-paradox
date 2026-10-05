# Product Price Optimiser

Predicts optimal product prices to maximise revenue based on historical sales data.


## Overview
- **Data Generation**: Synthetic dataset of `Base_Price` and `Sales`:
    - `Sales = Demand - 1.5 * Base_Price + noise`
    - `Revenue = Base_Price * Sales`

- **Model**: Linear Regression predicts sales from price.

- **Evaluation**: Mean Squared Error (MSE) measures model accuracy.

- **Optimisation**: Iterate over price range, compute revenue:
    - Optimal price = `argmax(price * predicted_sales)`

- **Visualisation**: Revenue curve with optimal price highlighted.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```