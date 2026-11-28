# Face Recognition System

A real-time face recognition system that learns identities locally and recognises them instantly using a `live` camera feed.


## Overview
- Face detection uses `Haar cascade classifiers` to locate facial regions.

- Each detected face is `resized` to a fixed resolution `(200 × 200)`.

- Recognition is performed using `LBPH (Local Binary Patterns Histogram)`:
    - For each pixel, compare neighbours with the centre pixel.
    - Encode comparisons as a binary pattern.
    - Build histograms over local regions.
    - Compare histograms using distance metrics.

- Prediction returns the closest matching identity and confidence score.

- All data is stored locally; no network access is required.


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```