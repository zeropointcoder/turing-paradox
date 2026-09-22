# Dog vs Cat Image Classifier

A computer vision system that distinguishes `dogs` from `cats` using `synthetic` image patterns and classical machine learning.


## Overview
- `Synthetic` grayscale images are generated programmatically to represent `dogs` and `cats`

- Each image is flattened into a feature `vector` 

- Features are standardised using:

  $$
  z = \frac{x - \mu}{\sigma}
  $$

- A linear Support Vector Machine learns a separating hyperplane:

  $$
  f(x) = w \cdot x + b
  $$

- The trained model predicts unseen images and accuracy is calculated on a hold-out set
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```