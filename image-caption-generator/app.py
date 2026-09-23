import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader 
from PIL import Image, ImageDraw


# Dataset
class SyntheticImageDataset(Dataset):
    def __init__(self, samples_per_class=100, image_size=64):
        self.image_size = image_size
        self.labels = ["square", "circle", "triangle"]
        self.data = []
        self.targets = []

        for idx, label in enumerate(self.labels):
            for _ in range(samples_per_class):
                img = self._draw_shape(label)
                self.data.append(img)
                self.targets.append(idx)

    def _draw_shape(self, label):
        img = Image.new("RGB", (self.image_size, self.image_size), "black")
        draw = ImageDraw.Draw(img)

        if label == "square":
            draw.rectangle([16, 16, 48, 48], fill="white")
        elif label == "circle":
            draw.ellipse([16, 16, 48, 48], fill="white")
        elif label == "triangle":
            draw.polygon([(32, 12), (12, 52), (52, 52)], fill="white")

        return np.array(img, dtype=np.float32) / 255.0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx]).permute(2, 0, 1)
        y = torch.tensor(self.targets[idx])
        return x, y


# Model
class CaptionCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# Trainer
class Trainer:
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

    def train(self, epochs=5):
        self.model.train()
        for _ in range(epochs):
            for x, y in self.loader:
                preds = self.model(x)
                loss = self.loss_fn(preds, y)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()


# Caption Generator
class CaptionGenerator:
    def __init__(self, model, labels):
        self.model = model
        self.labels = labels
        self.templates = [
            "A photo of a {}",
            "An image showing a {}",
            "A picture depicting a {}"
        ]

    def generate(self, image):
        self.model.eval()
        x = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            idx = torch.argmax(self.model(x)).item()
        template = self.templates[idx % len(self.templates)]
        return template.format(self.labels[idx])


# Streamlit App
st.title("Visual Caption Generator")
st.markdown("Image captioning using a trained neural network.")

# Load dataset and dataloader
dataset = SyntheticImageDataset()
loader = DataLoader(dataset, batch_size=10, shuffle=True)

# Train model (offline, quick for small dataset)
model = CaptionCNN(num_classes=len(dataset.labels))
trainer = Trainer(model, loader)
trainer.train()

# Initialise caption generator
captioner = CaptionGenerator(model, dataset.labels)

# Persist dropdown options in session_state
if "sample_options" not in st.session_state:
    st.session_state.sample_options = np.random.choice(
        len(dataset), size=20, replace=False
    ).tolist()

sample_idx = st.selectbox(
    "Select an image:",
    st.session_state.sample_options
)

# Display selected image
sample_img, _ = dataset[sample_idx]
st.image(
    np.transpose(sample_img.numpy(), (1, 2, 0)),
    caption="Selected Image",
    width=300
)

# Generate caption on button click
if st.button("Generate Caption"):
    caption = captioner.generate(np.transpose(sample_img.numpy(), (1, 2, 0)))
    st.success(f"Caption: {caption}")