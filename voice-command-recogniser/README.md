# Voice Command Recogniser

Voice command classifier using MFCC features and KNN on synthetic speech-like audio.


## Overview
- Generate synthetic audio dataset simulating voice commands.

- Extract `MFCC` features from each audio sample.

- Train a `K`-Nearest Neighbours classifier on the features.

- Test predictions on held-out samples with evaluation metrics.

- Command-line interface allows repeated testing.


## Run
```bash
pip install -r requirements.txt
```

```bash
python3 voice_command_recogniser.py
```