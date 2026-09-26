# Language Detection Model

A machine learning project that detects whether input text is written in UK English or a non-English language using statistical language modelling.


## Overview
- Text samples are generated locally with UK English spellings and common foreign language phrases

- Each sentence is transformed using `TF-IDF`:
  
    `TF-IDF(t, d) = TF(t, d) × log(N / DF(t))`

- Both `unigrams` and `bigrams` are used to capture spelling patterns

- A `Multinomial Naive Bayes` classifier estimates:
  
    `P(class | text) ∝ P(class) × ∏ P(word | class)`

- The trained model predicts the most probable language class
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```