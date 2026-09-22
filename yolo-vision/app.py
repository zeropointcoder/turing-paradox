import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw

DEVICE = "cpu"
IMG_SIZE = 64


class SyntheticDataset:
    def __init__(self, size=IMG_SIZE, samples=50):
        self.size = size
        self.samples = samples

    def generate(self):
        images, targets = [], []
        for _ in range(self.samples):
            img = Image.new("RGB", (self.size, self.size), "black")
            draw = ImageDraw.Draw(img)

            x1, y1 = np.random.randint(5, 30, 2)
            x2 = x1 + np.random.randint(10, 25)
            y2 = y1 + np.random.randint(10, 25)

            draw.rectangle([x1, y1, x2, y2], fill="white")

            images.append(np.array(img).transpose(2, 0, 1) / 255.0)
            targets.append([
                x1 / self.size,
                y1 / self.size,
                x2 / self.size,
                y2 / self.size
            ])

        return (
            torch.from_numpy(np.array(images)).float(),
            torch.from_numpy(np.array(targets)).float()
        )


class YOLOv5Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.head = nn.Linear(32 * 8 * 8, 4)

    def forward(self, x):
        x = self.backbone(x).view(x.size(0), -1)
        return self.head(x)


class Trainer:
    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.MSELoss()
        self.optim = torch.optim.Adam(model.parameters())

    def train(self, images, targets, epochs=2):
        self.model.train()
        for _ in range(epochs):
            preds = self.model(images)
            loss = self.loss_fn(preds, targets)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()


class Detector:
    def __init__(self, model):
        self.model = model.eval()

    def detect(self, image):
        tensor = (
            torch.tensor(image.transpose(2, 0, 1) / 255.0, dtype=torch.float32)
            .unsqueeze(0)
        )
        with torch.no_grad():
            return self.model(tensor)[0].numpy()


st.title("YOLOv5 Object Detection")

dataset = SyntheticDataset()
images, targets = dataset.generate()

model = YOLOv5Tiny().to(DEVICE)
trainer = Trainer(model)
trainer.train(images, targets)

detector = Detector(model)

index = st.slider("Sample image", 0, len(images) - 1, 0)
img = (images[index].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

box = detector.detect(img)
x1, y1, x2, y2 = box

x_min = int(min(x1, x2) * IMG_SIZE)
y_min = int(min(y1, y2) * IMG_SIZE)
x_max = int(max(x1, x2) * IMG_SIZE)
y_max = int(max(y1, y2) * IMG_SIZE)

x_min = max(0, min(IMG_SIZE - 1, x_min))
y_min = max(0, min(IMG_SIZE - 1, y_min))
x_max = max(0, min(IMG_SIZE - 1, x_max))
y_max = max(0, min(IMG_SIZE - 1, y_max))

pil_img = Image.fromarray(img)
draw = ImageDraw.Draw(pil_img)
draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

st.image(pil_img, caption="Detected object")