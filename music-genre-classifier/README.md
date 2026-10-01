# Music Genre Classifier

Classify music genres using synthetic features.


## Overview
- **Data Generation**: Generates random features for multiple music genres.

- **Standardisation**: Scales features using `StandardScaler`.

- **Model**: Random Forest classifier trained on generated data.

- **Prediction**: Model predicts genre labels for test data.

- **Evaluation**: Outputs accuracy and detailed classification report.

**Formulas used conceptually**:
- Standardisation:  `x_scaled = (x - μ) / σ`
- Random Forest prediction: `majority_vote(tree_predictions)`


## Run
```bash
pip install -r requirements.txt
```

```bash
python3 music_genre_classifier.py
```