# Stock Trend Predictor

A command-line application that predicts short-term stock price trends using simple linear regression on synthetic market data.


## Overview
- Generates offline synthetic stock prices using:
  - Linear trend component
  - Random `Gaussian` noise

- Normalises the time feature to improve numerical stability during training

- Applies simple linear regression:
  - Prediction formula:  
    **ŷ = w·x + b**

- Optimises parameters using gradient descent:
  - Weight update:  
    **w ← w − α·∂MSE/∂w**
  - Bias update:  
    **b ← b − α·∂MSE/∂b**

- Evaluates performance using Mean Squared Error `(MSE)`

- Determines overall price trend direction from the learned slope
 

## Run
```bash
pip install -r requirements.txt
```

```bash
python3 stock_trend_predictor.py
```