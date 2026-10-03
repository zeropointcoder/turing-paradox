# Movie Script Generator

A character-level LSTM model that generates movie script text, trained on a small corpus with live training feedback.


## Overview
- Load the script corpus and encode characters as `one-hot` vectors.

- Train a multi-layer `LSTM` to predict the `next` character in sequences.

- During training, detach hidden states each step to avoid backward graph errors.

- Show epoch-wise `loss`.

- Generate new scripts by `sampling` the LSTM one character at a time.


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py

### Output:
### The meaningless text is just a limitation of the tiny dataset and character-level LSTM.
```