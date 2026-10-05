# Medical Image Classifier

A neural network classifies synthetic medical images into normal or abnormal categories.


## Overview
- **Data Generation**: Creates `64x64` synthetic images:
    - Circles → Normal
    - Squares → Abnormal

- **Model Architecture**:
    - Scikit-learn MLP: `64→32` hidden layers
    - Softmax-like output

- **Training**:
    - CPU-only, offline
    - Loss implicitly handled by `MLPClassifier`

- **Evaluation**:
    - Reports Precision, Recall, `F1`-score

**Formulas**:
- Cross-entropy loss (conceptual):
    $$
    L = - \sum_i y_i \log(\hat{y}_i)
    $$


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```