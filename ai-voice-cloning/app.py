import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from pathlib import Path
from scipy.io.wavfile import write


# Data Layer
class VoiceDataset:
    def __init__(self, sample_rate=16000, n_mfcc=13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.features = []

    def load(self, folder_path):
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder {folder_path} not found")
        wav_files = list(folder.glob("*.wav"))
        if not wav_files:
            raise ValueError("No WAV files found in folder")
        for wav in wav_files:
            audio, sr = sf.read(wav)
            # Convert stereo to mono if needed
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            # Correct resample call
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=self.sample_rate)
            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=self.n_mfcc)
            self.features.append(mfcc.T)
        return np.vstack(self.features)


# ML Model
class VoiceAutoEncoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class VoiceModel:
    def __init__(self, feature_dim):
        self.model = VoiceAutoEncoder(feature_dim)
        self.trained = False
        self.feature_mean = None  # mean of encoded features for smoother synthesis

    def train(self, features, epochs=40):
        optimiser = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()
        x = torch.tensor(features, dtype=torch.float32)

        for _ in range(epochs):
            optimiser.zero_grad()
            recon = self.model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            optimiser.step()

        self.trained = True
        # store mean of encoded features for smoother output
        with torch.no_grad():
            encoded = self.model.encoder(x)
            self.feature_mean = torch.mean(encoded, dim=0)

    def synthesise(self, text_length):
        if not self.trained:
            raise RuntimeError("Model not trained")

        # Generate latent vectors near the mean for smoother output
        latent = self.feature_mean + 0.1 * torch.randn(text_length * 10, 16)

        with torch.no_grad():
            mfcc = self.model.decoder(latent).numpy().T

        # Simple smoothing across MFCC frames
        mfcc_smooth = np.copy(mfcc)
        for i in range(1, mfcc.shape[1]):
            mfcc_smooth[:, i] = 0.8 * mfcc_smooth[:, i-1] + 0.2 * mfcc[:, i]

        # Convert MFCC back to waveform
        audio = librosa.feature.inverse.mfcc_to_audio(mfcc_smooth)
        return audio.astype(np.float32)


# Evaluation
class VoiceEvaluator:
    def score(self, real_features, synthetic_features):
        min_len = min(len(real_features), len(synthetic_features))
        return float(np.mean((real_features[:min_len] - synthetic_features[:min_len]) ** 2))


# Streamlit UI
st.title("Voice Replicator (ML)")
st.markdown("Trains a voice prototype model from WAV samples and generates smoother speech-like audio.")

folder = st.text_input(
    "Folder containing WAV samples", 
    placeholder="Example: voice-samples"
)

if st.button("Train Model"):
    dataset = VoiceDataset()
    try:
        features = dataset.load(folder)
        model = VoiceModel(features.shape[1])
        model.train(features)
        st.session_state["model"] = model
        st.session_state["features"] = features
        st.success("Model trained successfully!")
    except Exception as e:
        st.error(str(e))

text = st.text_input(
    "Text to synthesise", 
    placeholder="Example: Hello, this is my synthetic voice."
)

if st.button("Generate Voice"):
    if "model" not in st.session_state:
        st.warning("Train the model first")
    else:
        model = st.session_state["model"]
        audio = model.synthesise(len(text))
        write("synthesised.wav", 16000, audio)
        st.audio("synthesised.wav")