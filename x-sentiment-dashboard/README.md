# Twitter Sentiment Dashboard

An interactive dashboard that analyses sentiment trends from UK-style Twitter messages using classical machine learning.


## Overview
- A small UK-English tweet dataset is generated locally and loaded into memory

- Text is transformed into numerical features using **TF-IDF**:
  
    $$
    \text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)
    $$

- A **Logistic Regression** classifier estimates sentiment probabilities:
  
    $$
    P(y=k|x) = \frac{e^{w_k x}}{\sum_j e^{w_j x}}
    $$

- Predictions are visualised interactively using Streamlit


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```