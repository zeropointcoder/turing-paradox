# Pretrain Transformer

A transformer is pretrained from scratch on a generated UK-English corpus to learn basic language structure.


## Overview
- A synthetic UK-English text corpus is generated locally

- Text is tokenised at `character level`

- Input sequences of length `n` predict the next character

- Transformer encoder learns contextual representations

- Training minimises categorical `cross-entropy` loss

**Core formulation**

- Token embedding:  
    `E = Embedding(x) + PositionalEncoding`

- Self-attention:  
    `Attention(Q,K,V) = softmax(QKᵀ / √d) V`

- Objective:  
    `L = −Σ y log(ŷ)`
 

## Run
```bash
pip install -r requirements.txt
```

```bash
python3 train_transformer.py --epochs 10
```