# Handwritten digit recognition

An offline handwritten digit recognition application with interactive predictions using a lightweight neural model.


## Overview
- Uses a built-in handwritten digit dataset (`8×8` grayscale images)

- Each image is flattened into a `64-dimensional` vector

- A `feedforward` neural network learns digit patterns

- Prediction is the class with `maximum` output probability
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```