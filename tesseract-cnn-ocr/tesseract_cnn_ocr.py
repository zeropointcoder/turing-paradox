import argparse
import string
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional Tesseract (runtime-safe)
try:
    import pytesseract
    from pytesseract import TesseractNotFoundError
    TESS_AVAILABLE = True
except Exception:
    TESS_AVAILABLE = False


class SyntheticTextDataset(Dataset):
    def __init__(self, samples=400):
        self.chars = string.ascii_uppercase
        self.samples = samples

    def __len__(self):
        return self.samples

    def render(self, char):
        img = Image.new("L", (28, 28), 255)
        draw = ImageDraw.Draw(img)
        draw.text((6, 4), char, fill=0)
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr

    def __getitem__(self, idx):
        char = np.random.choice(list(self.chars))
        x = self.render(char)
        y = self.chars.index(char)
        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), y


class OCRNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 11 * 11, classes)
        )

    def forward(self, x):
        return self.net(x)


class Trainer:
    def __init__(self, model):
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(model.parameters())

    def train(self, loader, epochs=25):
        self.model.train()
        for _ in range(epochs):
            for x, y in loader:
                self.optim.zero_grad()
                out = self.model(x)
                loss = self.loss(out, y)
                loss.backward()
                self.optim.step()


class CharacterSegmenter:
    def segment(self, image):
        img = image.convert("L")
        arr = np.array(img)
        binary = arr < 200

        cols = binary.sum(axis=0)
        cuts = np.where(cols > 0)[0]

        segments = []
        if len(cuts) == 0:
            return segments

        start = cuts[0]
        for i in range(1, len(cuts)):
            if cuts[i] != cuts[i - 1] + 1:
                segments.append((start, cuts[i - 1]))
                start = cuts[i]
        segments.append((start, cuts[-1]))

        chars = []
        for s, e in segments:
            char_img = img.crop((s, 0, e + 1, img.height))
            chars.append(char_img)

        return chars


class OCRPipeline:
    def __init__(self):
        self.chars = string.ascii_uppercase
        self.model = OCRNet(len(self.chars))

        dataset = SyntheticTextDataset()
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        Trainer(self.model).train(loader)

        self.segmenter = CharacterSegmenter()

    def cnn_predict(self, image):
        img = image.resize((28, 28)).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        t = torch.tensor(arr).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            idx = self.model(t).argmax(1).item()

        return self.chars[idx]

    def run(self, image_path):
        image = Image.open(image_path)

        if TESS_AVAILABLE:
            try:
                text = pytesseract.image_to_string(image, lang="eng").strip()
                if text:
                    return text
            except (TesseractNotFoundError, RuntimeError, Exception):
                pass

        chars = self.segmenter.segment(image)
        if not chars:
            return ""

        return "".join(self.cnn_predict(c) for c in chars)


def generate_sample_image(path="sample.png"):
    img = Image.new("L", (160, 50), 255)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "HELLO", fill=0)
    img.save(path)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Path to image")
    args = parser.parse_args()

    img_path = args.image if args.image else generate_sample_image()
    print("\n", OCRPipeline().run(img_path), "\n")