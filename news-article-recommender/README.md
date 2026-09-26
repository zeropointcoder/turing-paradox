# News Article Recommender

A system that recommends UK news articles based on semantic similarity to user input.


## Overview
- Articles are represented as `TF-IDF` vectors

- A user query is converted into the same vector space

- `Cosine similarity` measures relevance between query and articles

    `similarity(a, b) = (a · b) / (||a|| × ||b||)`

- Articles are ranked by `similarity` score

- Top matches are displayed instantly


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```