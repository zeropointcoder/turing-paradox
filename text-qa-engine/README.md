# Text QA Engine

A question answering system that retrieves the most relevant response from a local knowledge base using statistical text similarity.


## Overview
- Text documents are converted into numerical vectors using `TF–IDF`  
    - Term Frequency:  
        $$
        \text{tf}(t, d) = \frac{\text{count of } t \text{ in } d}{\text{total terms in } d}
        $$
    - Inverse Document Frequency:  
        $$
        \text{idf}(t) = \log\left(\frac{N}{1 + n_t}\right)
        $$

- User questions are transformed using the same vector space

- `Cosine similarity` measures closeness between vectors  
    $$
    \cos(\theta) = \frac{A \cdot B}{\|A\|\|B\|}
    $$

- The document with the highest similarity score is returned as the answer

- A confidence level is derived from the similarity score
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```