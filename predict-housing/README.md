#  Predict Housing

An interactive Streamlit application that predicts UK housing prices using linear regression on generated property data.


## Overview
- A synthetic housing dataset is generated entirely offline

- Features:
    - Property size `(m²)`
    - Number of bedrooms
    - Property age

- `Linear` regression learns the relationship:

  **ŷ = β₀ + β₁x₁ + β₂x₂ + β₃x₃**

- Model performance is evaluated using:
  - Root Mean Squared Error `(RMSE)`
  - `R²` score

- Users can input property details to get an instant price estimate


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```