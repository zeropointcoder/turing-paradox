# CIFAR-10 CNN Image Classification

A compact convolutional neural network that performs `CIFAR-10–style` image classification using a fully offline synthetic dataset.

**NOTE**: "Image classification" ≠ "image visualization"

## Overview
- Images are generated offline as `32×32` RGB tensors 

- A convolutional neural network extracts spatial features  

- Feature maps are downsampled using `max` pooling  

- Final classification uses a `softmax-based` linear layer  

**Core operations**

- Convolution  
    `y = x * w + b`

- Activation `(ReLU)`  
    `f(x) = max(0, x)`

- Loss function `(Cross-Entropy)`  
    `L = −∑ y log(ŷ)`

- Optimisation: `Adam` gradient descent
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```