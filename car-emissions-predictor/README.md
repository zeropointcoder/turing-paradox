# Car CO₂ Emissions Predictor

Predict CO₂ emissions for cars based on engine specifications and fuel consumption.


## Overview
- Dataset is `generated offline` using random values for `engine size`, `cylinders`, and `fuel consumption`.

- `Linear Regression model` predicts `CO₂` emissions.

- **Formula Concept**:
    - `CO₂ ≈ 120 + (30 × Engine Size) + (5 × Cylinders) + noise`

- Model evaluated using:
    - Mean Squared Error `(MSE)`
    - `R²` Score


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```