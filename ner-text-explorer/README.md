# NER Text Explorer

UK-English Named Entity Recognition app using simple ML models for Person, Organisation, and Location.


## Overview
- Synthetic `UK`-English sentences with annotated entities are generated.

- Each entity type (`PERSON`, `LOCATION`, `ORG`) gets a binary classifier using `TF-IDF` features.

- User input text is transformed and passed to classifiers to predict entities.

- Streamlit app highlights the detected entities.


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py

### The custom text option only works if the words you type match the dictionaries in PERSONS, LOCATIONS, ORGANISATIONS
```