# Personalised Recommender Using Embeddings

Suggests items to users based on similarity between their preferences and item features.


## Overview
- Generate an item-feature matrix representing each item as a feature vector.

- Scale features using `StandardScaler`:
    - $$z = \frac{x - \mu}{\sigma}$$

- Pre-generate a unique user preference vector for each user.

- Compute `cosine` similarity between user vector and item embeddings:
    - $$\text{similarity} = \frac{u \cdot i}{\|u\| \, \|i\|}$$

- Recommend `top-k` items with highest similarity.


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```