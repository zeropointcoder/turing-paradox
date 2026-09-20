# Chatbot NLP (British)

A UK-English chatbot that uses ML for intent-based responses.


## Overview
- Normalises text (`lowercase`, `strip` spaces).

- Converts text to `TF-IDF` features.

- Uses a scikit-learn classifier `(MultinomialNB)` trained on small offline intent dataset.

- Predicts user intent and returns the corresponding `UK`-English response.

- UI maintains chat history.


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```