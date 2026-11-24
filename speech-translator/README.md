# Multi-language Speech Translator

Multi-language speech translator with ML-based language detection, word translation, and UK English output.


## Overview
- Train a `Bayes` classifier on sample sentences for language detection.

- Detect the language of the input transcript using the trained classifier.

- Translate all words in the sentence using a dictionary enhanced with ML-based word mapping.

- Normalise the translated output to `UK` English spelling.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```