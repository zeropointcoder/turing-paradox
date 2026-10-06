# Text Translator

Implement ML-based US→UK English spelling converter.


## Overview
- Generate training data from `US→UK` vocabulary and simple pattern rules.

- Convert words into `character`-level features (`n`-grams).

- Train an `MLP` classifier to predict `UK` spellings.

- Use trained model to translate input text word by word.

- Display results in real-time.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```