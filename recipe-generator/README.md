# AI-Powered Recipe Generator

An AI-enhanced recipe generator using a PyTorch LSTM model for coherent cooking instructions.


## Overview
- Tokenises the corpus and creates sequences for training.

- Trains a small `LSTM` network to learn word transitions.

- Generates recipes word-by-word based on a `start` word.


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py

### Small Corpus limitation: Sentences can abruptly jump.
### Small Corpus limitation: Words can repeat awkwardly.
### Small Corpus limitation: Some output looks meaningless.
```