# Image Transformer

An ML project that colourises user-drawn or uploaded sketches using a CNN trained on synthetic shapes.


## Overview
- Generates a dataset of random coloured shapes and corresponding sketches.

- Trains a CNN on these synthetic sketches.

- User can draw on a canvas or upload sketches.

- Model predicts a colourised version of the sketch.

- The output may look grey or blank at first because the model is trained on a small set of simple synthetic shapes and is still learning colour patterns. This is normal and shows the network is working internally, not a bug.

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```