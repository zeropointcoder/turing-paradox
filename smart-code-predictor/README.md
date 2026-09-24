# Smart Code Predictor

Predicts the next few lines of Python code using a `character-level RNN`.


## Overview
- **Dataset Generation**: Generates small offline Python code snippets.

- **Character-level RNN**:
  - Embeds characters → passes through `GRU` → outputs probability distribution of next character.
  - Loss: `CrossEntropyLoss`
  - Optimiser: `Adam`

- **Prediction**:
  1. Take starting sequence.
  2. Encode chars → feed to model → sample next char.
  3. Append and repeat for desired length.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```