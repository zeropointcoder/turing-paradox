import streamlit as st
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


# Model Classes
class SimpleSegNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(SimpleSegNet, self).__init__()
        self.enc1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dec1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = nn.Conv2d(16, out_channels, 3, padding=1)
    
    def forward(self, x):
        x1 = F.relu(self.enc1(x))
        x2 = F.relu(self.enc2(self.pool(x1)))
        x3 = F.relu(self.dec1(x2))
        x4 = self.dec2(x3)
        return x4


# Data Generation
class SceneDataset:
    def __init__(self, size=(128, 128)):
        self.size = size

    def generate(self):
        """Generates a synthetic urban scene image and segmentation mask"""
        img = np.zeros((*self.size, 3), dtype=np.uint8)
        mask = np.zeros((*self.size, 3), dtype=np.uint8)

        # Sky
        img[:64, :, :] = [135, 206, 235]  # light blue
        mask[:64, :, 0] = 1

        # Buildings
        img[64:96, :, :] = [169, 169, 169]  # grey
        mask[64:96, :, 1] = 1

        # Road
        img[96:, :, :] = [128, 128, 128]  # dark grey
        mask[96:, :, 2] = 1

        return img, mask


# Streamlit Interface
st.title("Urban Scene Segmentation")

st.markdown("""
This tool segments synthetic urban scenes into Sky, Buildings, and Road using a lightweight neural network.
""")

dataset = SceneDataset()
img, mask = dataset.generate()

st.subheader("Original Scene")
st.image(img, width=400)

# Convert to tensor for model
img_tensor = torch.tensor(img / 255.0, dtype=torch.float32).permute(2,0,1).unsqueeze(0)

model = SimpleSegNet()
model.eval()
with torch.no_grad():
    pred = model(img_tensor)
    pred_mask = torch.argmax(pred, dim=1).squeeze().numpy()

# Map predictions to colours
colours = np.array([[135,206,235], [169,169,169], [128,128,128]], dtype=np.uint8)
seg_image = colours[pred_mask]

st.subheader("Segmented Scene")
st.image(seg_image, width=400)
