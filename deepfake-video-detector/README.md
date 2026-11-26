# Deepfake Video Detector

Deepfake video detector using a CNN trained on video frames.


## Overview
- Load a video and `extract` frames.

- Preprocess frames (`resize`, `normalise`).

- Feed frames into a small convolutional neural network `(CNN)` to predict probability of deepfake.

- Aggregate `frame`-level predictions to give a `video`-level prediction.

- Display `prediction` and `confidence`.


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```