# Object Tracking in Video

Generate synthetic moving objects, detect them, and use an ML tracker to track them in real-time.


## Overview
- Make moving circles → `SyntheticVideo` creates objects that move around.

- Find their centers → `ObjectDetector` detects where the circles are.

- Move rings toward them → `MLTracker` predicts positions and moves blue rings to follow.


## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```