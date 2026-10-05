# AI Resume Analyser with NLP Suggestions

ML-powered Resume Analyser scoring resumes on action verb usage and keyword relevance.


## Overview
- Clean and tokenise resume text.

- Convert text into `TF-IDF` features.

- Train an ML model on synthetic labeled data (high vs low action verb resumes).

- Predict a resume `score` and provide actionable insights (`spelling`, `UK` spelling, keyword `density`).

- Display results.


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```