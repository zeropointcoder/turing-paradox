# AI Email Responder

A machine-learning email intent classifier that generates context-appropriate email responses using trained NLP models.


## Overview
- Labelled emails are embedded using `TF-IDF`

- A classifier learns email intent from text

- New emails are classified at runtime

- Responses are generated based on predicted intent
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py

### Based on a `label`'s category, the `response` is generated as output 
```