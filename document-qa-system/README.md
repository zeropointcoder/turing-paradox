# Document Q&A System

A local document question-answering system that retrieves and surfaces relevant passages using vector similarity.


## Overview
- Documents are converted into `TF-IDF` vectors
    `TF-IDF(w, d) = TF(w, d) × log(N / DF(w))`

- User queries are vectorised using the same feature space

- `Cosine` similarity ranks documents by semantic closeness  
    `cos(θ) = (A · B) / (||A|| ||B||)`

- Top-ranked passages are retrieved

- The highest-scoring passage is returned as the `answer`


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```