import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image


class ImageLoader:
    def __init__(self, size=128):
        self.size = size

    def load(self, image_file):
        image = Image.open(image_file).convert("RGB")
        image = image.resize((self.size, self.size))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return image


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layers(x)


class StyleTransferTrainer:
    def __init__(self, model):
        self.model = model

    def gram_matrix(self, features):
        b, c, h, w = features.size()
        f = features.view(c, h * w)
        return torch.mm(f, f.t()) / (c * h * w)

    def train(self, content, style, steps=100, lr=0.03):
        target = content.clone().requires_grad_(True)
        optimiser = optim.Adam([target], lr=lr)

        with torch.no_grad():
            style_features = self.model(style)
            style_gram = self.gram_matrix(style_features)

        for _ in range(steps):
            optimiser.zero_grad()
            target_features = self.model(target)

            content_loss = torch.mean((target_features - self.model(content)) ** 2)
            style_loss = torch.mean((self.gram_matrix(target_features) - style_gram) ** 2)

            loss = content_loss + 200 * style_loss # changing 200 to another value gives varying result
            loss.backward()
            optimiser.step()

        return target.detach()


class ImageEvaluator:
    def tensor_to_image(self, tensor):
        image = tensor.squeeze(0).permute(1, 2, 0).numpy()
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(image)


st.set_page_config(page_title="Image Style Transfer", layout="centered")
st.title("Image Style Transfer")

loader = ImageLoader()
model = SimpleCNN()
trainer = StyleTransferTrainer(model)
evaluator = ImageEvaluator()

content_file = st.file_uploader("Upload content image", type=["jpg", "png"])
style_file = st.file_uploader("Upload style image", type=["jpg", "png"])

if content_file and style_file:
    content = loader.load(content_file)
    style = loader.load(style_file)

    st.subheader("Input Images")
    col1, col2 = st.columns(2)
    col1.image(content_file, caption="Content", width=250)
    col2.image(style_file, caption="Style", width=250)

    if st.button("Run Style Transfer"):
        with st.spinner("Processing..."):
            output = trainer.train(content, style)
            result = evaluator.tensor_to_image(output)

        st.subheader("Stylised Output")

        st.image(result, width=300)

        st.html("Style is applied, but too weakly")
        st.html("Random, shallow networks preserve content by default")
        st.html("Increase style weight or start from noise to see visible change")