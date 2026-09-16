# Facial Expression Detector

A machine learning system that identifies facial expressions from generated face images using classical techniques.


## Overview
- Synthetic face images are generated using `geometric primitives`.

- Each image is converted into a grayscale vector `x ∈ R^1024`.

- Dimensionality reduction is applied using `PCA`:

  $$
  Z = XW
  $$

  where `(W)` contains the principal components.

- A `Support Vector Machine` learns a non-linear decision boundary:

  $$
  f(x) = \text{sign} \left( \sum_i \alpha_i K(x_i, x) + b \right)
  $$

- The trained model predicts one of `three` expressions:
    - Happy
    - Sad
    - Angry
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```