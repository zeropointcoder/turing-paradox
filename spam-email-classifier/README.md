# Spam Email Classifier

An offline machine-learning application that classifies emails as spam or legitimate using traditional natural language processing techniques.


## Overview
- Emails are converted into numerical vectors using **TF-IDF**

- A **Multinomial Naive Bayes** classifier learns word probabilities

- Prediction is based on maximum posterior probability:

    `P(class | email) ∝ P(class) × Π P(word | class)`



## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```