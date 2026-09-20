# Text-to-Speech Synthesis

UK-English text-to-speech with ML-based spelling normalisation and duration prediction.


## Overview
- UKTextNormaliser – `ML` model predicts `UK` spellings from input text.

- SpeechEvaluator – ML model predicts approximate speech `duration`.

- SpeechModel – `TTS` using `pyttsx3`.

- SpeechController – Combines all components.

- Interactive text input and speech output.
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```