import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


# Video Loader
class VideoLoader:
    def __init__(self, video_path):
        self.video_path = video_path
        self.frames = []

    def load_frames(self, max_frames=30):
        cap = cv2.VideoCapture(str(self.video_path))
        count = 0
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (64, 64))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0  # normalise
            self.frames.append(frame.astype(np.float32))
            count += 1
        cap.release()
        self.frames = np.array(self.frames)
        return self.frames


# CNN Model
class DeepfakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # Deepfake or Authentic

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Deepfake Detector
class DeepfakeDetector:
    def __init__(self):
        self.model = DeepfakeCNN()
        self.model.eval()  # evaluation mode

    def predict(self, frames):
        if len(frames) == 0:
            return "No frames found", 0.0
        frames_tensor = torch.tensor(frames).permute(0, 3, 1, 2)  # batch, channel, H, W
        with torch.no_grad():
            outputs = self.model(frames_tensor)
            probs = F.softmax(outputs, dim=1)[:, 1]  # probability of Deepfake
            score = probs.mean().item()
        label = "Deepfake" if score > 0.5 else "Authentic"
        return label, score


# Streamlit Interface
st.set_page_config(page_title="Deepfake Video Detector")
st.title("Deepfake Video Detector (ML)")

uploaded_file = st.file_uploader("Upload a short video (MP4 preferred)", type=["mp4", "avi"])

if uploaded_file:
    with st.spinner("Processing video..."):
        video_path = Path("temp_video.mp4")
        video_path.write_bytes(uploaded_file.getvalue())

        loader = VideoLoader(video_path)
        frames = loader.load_frames()

        detector = DeepfakeDetector()
        label, confidence = detector.predict(frames)

        st.video(uploaded_file)
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence Score:** {confidence:.2f}")
