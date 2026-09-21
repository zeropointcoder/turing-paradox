# Vision Calculator

An ML-based calculator that recognises printed arithmetic expressions using augmented templates and evaluates them accurately.


## Overview
- Generate synthetic templates for digits and symbols `(0–9, +, -, *, /, (, ))`.

- Augment templates with small `rotations`, `shifts`, and `scaling` to improve generalisation.

- Flatten images and train a `Random Forest classifier` on the augmented dataset.

- Accept user input via Streamlit: typed text or uploaded printed images.

- Segment individual characters from the input image using `contour` detection.

- Predict each character with the trained `ML` model.

- `Combine` characters into an arithmetic expression.

- Evaluate the expression safely and display the result along with the input image.


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py

### Output may be incorrect due to limited synthetic training data.
### Accuracy is constrained by basic character segmentation and model choice.
```