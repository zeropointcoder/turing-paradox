# Resume Screening Model

A machine-learning system that ranks CVs against a job description using semantic similarity.


## Overview
- Each CV and the job description are represented as text documents.

- Text is converted into numerical vectors using **TF-IDF**:

  **TF-IDF(t, d) = TF(t, d) × log(N / DF(t))**

- Similarity between a CV and the job description is computed using **cosine similarity**:

  **cos(θ) = (A · B) / (||A|| × ||B||)**

- CVs are ranked in descending order of `similarity` score.

- All data is generated locally and processed fully offline. 
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```