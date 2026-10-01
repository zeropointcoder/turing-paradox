# Hand Gesture Recognition

A hand gesture recognition system using synthetic landmark data and classical machine learning.


## Overview
- Each hand gesture is represented as `21 hand landmarks`

- Each landmark has `(x, y)` coordinates

- Feature vector size:

    `21 landmarks × 2 coordinates = 42 features`

- Synthetic landmark patterns are generated for:
    - Open hand
    - Fist
    - Peace sign

- A `k-Nearest Neighbours (KNN)` classifier is trained:

    `d(x, y) = √Σ(xᵢ − yᵢ)²`

- Prediction is based on the most common label among nearest samples
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```