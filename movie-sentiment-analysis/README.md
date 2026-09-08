# Movie Review Sentiment Analysis

An offline machine-learning application that analyses the sentiment of movie reviews using classical natural language processing techniques.


## Overview
- Text is converted into numerical features using **TF-IDF (1-grams & 2-grams)**
  - `TF-IDF = Term Frequency × Inverse Document Frequency`

- A **Logistic Regression** classifier with balanced class weights learns sentiment `polarity`
  - `σ(z) = 1 / (1 + e⁻ᶻ)`

- Model is trained on a small embedded dataset

- Predictions are made locally with no network access


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```