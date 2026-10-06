# Music Generation with RNN

Generates melodies using a recurrent neural network implemented in NumPy and outputs them as playable MIDI files.


## Overview
- Synthetic dataset of note sequences is generated offline.
    - Notes are integers representing pitches `(0–49)`
    - Each sequence has `sequence_length` notes

- Model:
    - Vanilla `RNN` implemented using NumPy
    - Hidden state updated at each timestep:
      - `h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)`
    - Output layer predicts next note probabilities:
      - `y_t = softmax(W_hy * h_t + b_y)`

- Training:
    - Simple gradient descent on output layer weights `(Why)` and bias `(b_y)`
    - Loss is categorical cross-entropy:
      - `L = - Σ y_true * log(y_pred)`

- Melody generation:
    - `Seed` notes initialise the hidden state
    - `Next` notes are predicted iteratively using the `RNN` equations

- `MIDI` conversion:
    - Predicted integer sequence mapped to `MIDI` note numbers `(60–109)`
    - Each note added as `note_on` and `note_off` events using `mido`
    - Streamlit provides a `downloadable MIDI` file
 

## Run
```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py
```