import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageLoader:
    def load(self, uploaded_file):
        image = Image.open(uploaded_file).convert("RGB")
        return np.array(image)


class ImagePreprocessor:
    def resize_and_normalise(self, image, size=(128,128)):
        img = Image.fromarray(image).resize(size)
        img_array = np.array(img).astype(np.float32)/255.0
        img_tensor = torch.from_numpy(img_array).permute(2,0,1).unsqueeze(0)
        return img_tensor

    def tensor_to_image(self, tensor):
        img_array = tensor.squeeze(0).permute(1,2,0).detach().numpy()
        img_array = (img_array*255).clip(0,255).astype(np.uint8)
        return img_array


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(3, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        # Decoder
        self.dec1 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec2 = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.dec1(x))
        x = torch.sigmoid(self.dec2(x))
        return x


class ImageEnhancer:
    def __init__(self):
        self.model = AutoEncoder()
        # Use simple offline training
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

    def enhance(self, img_tensor, steps=50):
        # Train autoencoder on this single image (self-supervised)
        self.model.train()
        target = img_tensor.clone()
        for _ in range(steps):
            self.optimiser.zero_grad()
            output = self.model(img_tensor)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimiser.step()
        self.model.eval()
        with torch.no_grad():
            return self.model(img_tensor)


class ImageEvaluator:
    def estimate_sharpness(self, image):
        import cv2
        return cv2.Laplacian(image, cv2.CV_64F).var()


st.set_page_config(page_title="ML Photo Enhancer", layout="centered")
st.title("ML Photo Enhancer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file:
    loader = ImageLoader()
    original = loader.load(uploaded_file)

    strength = st.slider("Enhancement strength", 1, 100, 50)

    preprocessor = ImagePreprocessor()
    img_tensor = preprocessor.resize_and_normalise(original)

    enhancer = ImageEnhancer()
    enhanced_tensor = enhancer.enhance(img_tensor, steps=strength)

    enhanced = preprocessor.tensor_to_image(enhanced_tensor)

    evaluator = ImageEvaluator()
    before_score = evaluator.estimate_sharpness(original)
    after_score = evaluator.estimate_sharpness(enhanced)

    col1, col2 = st.columns(2)
    col1.image(original, caption="Original", width=300)
    col2.image(enhanced, caption="Enhanced", width=300)

    st.markdown(
        f"**Sharpness score** - Before: `{before_score:.2f}` | After: `{after_score:.2f}`"
    )