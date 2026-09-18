# Open-source Framework - Linear Regression Framework

A linear regression project that trains and evaluates on synthetic data with visualisation and proper train/test split.


## Overview
- Generate synthetic data with `noise`.

- Split the data into `train` and `test` sets.

- Initialise a `linear regression model`.

- Train the model using `gradient descent` on training data.

- Evaluate model performance on test data with `mean squared error`.

- Visualise `predictions` against `true` values.


## Run
```bash
pip install -r requirements.txt
```

```bash
python3 linear_regression_framework.py
python3 linear_regression_framework.py --samples 1000 --epochs 1500
```