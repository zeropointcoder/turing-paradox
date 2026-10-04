# Keyword Extraction Tool

Keyword extraction and text classification tool with prediction confidence and keyword importance scoring.


## Overview
- User enters text into the app.

- Text is cleaned and preprocessed.

- `TF-IDF` extracts top keywords with relative importance (weights).

- TF-IDF vector is fed to a `Logistic Regression classifier`.

- App outputs top keywords with `weights`, `predicted category`, and `prediction confidence`.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```