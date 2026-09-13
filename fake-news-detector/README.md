# Fake News Detector

An offline machine learning application that classifies news text as real or fake using linguistic patterns.


## Overview
- Text is converted into numerical form using **TF–IDF**:
  
    `TF-IDF(w, d) = TF(w, d) × log(N / DF(w))`

- `Unigrams` and `bigrams` capture short contextual phrases.

- A **logistic regression classifier** learns probability boundaries:

    `P(y=1|x) = 1 / (1 + e^(-w·x))`

- The model is trained on a small, fully offline dataset.

- User input is transformed and classified instantly in the browser. 
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```