# AI Photo Enhancer

ML-based photo enhancer that improves image sharpness and contrast using a trainable neural network.


## Overview
- Load Image – User uploads a photo.

- Preprocess – Image resized and normalised for the model.

- Enhance via ML – A PyTorch convolutional autoencoder predicts an enhanced image.

- Evaluate – `Sharpness` of original vs enhanced measured using `Laplacian` variance.

- Display – Show original and enhanced images side by side.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```