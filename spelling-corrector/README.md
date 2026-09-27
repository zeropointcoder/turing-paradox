# Spelling Corrector

A UK English spelling corrector using a simple probabilistic unigram language model with edit-distance candidate generation.


## Overview
- Load `UK` corpus and build a vocabulary.

- Compute `unigram` probabilities for words from the corpus.

- Generate candidate corrections using edit-distance `1` and `2`.

- Select the most probable word based on corpus frequencies.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py

### Enter words with similar spelling as present in the dataset, e.g., color, organize, center. 
### The predictor corrects them with corresponding UK spelling.
```