import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader


DEVICE = "cpu"
IMAGE_SIZE = 32
NUM_CLASSES = 10
CLASS_NAMES = [
    "aeroplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]


class SyntheticCIFAR10(Dataset):
    def __init__(self, samples=800):
        self.x = np.random.rand(samples, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
        self.y = np.random.randint(0, NUM_CLASSES, samples)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class Trainer:
    def __init__(self, model):
        self.model = model
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimiser = optim.Adam(model.parameters())

    def train(self, loader, epochs):
        self.model.train()
        history = []
        for _ in range(epochs):
            total_loss = 0.0
            for x, y in loader:
                self.optimiser.zero_grad()
                loss = self.loss_fn(self.model(x), y)
                loss.backward()
                self.optimiser.step()
                total_loss += loss.item()
            history.append(total_loss / len(loader))
        return history


class Evaluator:
    def __init__(self, model):
        self.model = model

    def accuracy(self, loader):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                preds = self.model(x).argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        return correct / total


st.set_page_config(page_title="CIFAR-10 CNN (Offline)", layout="centered")
st.title("Offline CIFAR-10 Image Classification")

epochs = st.slider("Training epochs", 1, 10, 3)

if st.button("Train model"):
    dataset = SyntheticCIFAR10()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = CNNClassifier().to(DEVICE)
    trainer = Trainer(model)
    evaluator = Evaluator(model)

    losses = trainer.train(loader, epochs)
    acc = evaluator.accuracy(loader)

    st.success("Training completed")
    st.write(f"Final accuracy: **{acc:.2%}**")
    st.line_chart(losses)