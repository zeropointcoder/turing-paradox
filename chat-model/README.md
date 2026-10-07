# Chat Model

A Transformer-based conversational model trained on synthetic British English dialogue.


## Overview
- Uses a `word-level` vocabulary built entirely offline

- Transformer encoder stack with self-attention  
    `Attention(Q,K,V) = softmax(QKᵀ / √d) V`

- Trained using next-token prediction  
    `L = CrossEntropy(ŷₜ, yₜ₊₁)`

- Sampling uses temperature scaling and `top-k` filtering

- Dataset is generated locally with `UK` spelling and phrasing

- No external models, tokenisers, or downloads
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```