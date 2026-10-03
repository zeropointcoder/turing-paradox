import numpy as np
from scipy.io.wavfile import write
import os


os.makedirs("voice-samples", exist_ok=True)
sr = 16000  # sample rate (16 kHz)

# Tiny "UK English voice" samples (sine waves to mimic speech pitch)
frequencies = [220, 250, 280]  # slightly different for variety
texts = ["Hello there!", "Good morning!", "Testing voice!"]

for i, freq in enumerate(frequencies):
    t = np.linspace(0, 1, sr)
    waveform = 0.1 * np.sin(2 * np.pi * freq * t)  # low volume sine wave
    write(f"voice-samples/sample_{i+1}.wav", sr, waveform.astype(np.float32))

