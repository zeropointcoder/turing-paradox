# AI Voice Cloning

ML voice prototype generator that learns your voice style and produces smoother speech-like audio from WAV samples.


## Overview
- Load WAV samples and extract `MFCC` features.

- Train an `autoencoder` to learn voice characteristics.

- Generate latent vectors near the learned distribution.

- Smooth the MFCCs and reconstruct waveform.

- Play the generated `speech`-like audio.


## Run
```bash
pip install -r requirements.txt
```

```bash
### Create voice-samples first if they are not present in voice-samples/
python3 voice_sample.py

streamlit run app.py
```