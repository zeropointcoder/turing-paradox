import numpy as np
from PIL import Image, ImageDraw
import streamlit as st
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SyntheticPetDataset:
    def __init__(self, image_size=32, samples_per_class=200):
        self.image_size = image_size
        self.samples_per_class = samples_per_class

    def _dog_image(self):
        img = Image.new("L", (self.image_size, self.image_size), 0)
        d = ImageDraw.Draw(img)
        d.rectangle([6, 10, 26, 22], fill=255)
        d.rectangle([2, 6, 10, 14], fill=180)
        d.rectangle([22, 6, 30, 14], fill=180)
        return np.array(img)

    def _cat_image(self):
        img = Image.new("L", (self.image_size, self.image_size), 0)
        d = ImageDraw.Draw(img)
        d.ellipse([8, 10, 24, 26], fill=255)
        d.polygon([8, 10, 12, 2, 16, 10], fill=200)
        d.polygon([16, 10, 20, 2, 24, 10], fill=200)
        return np.array(img)

    def load(self):
        X, y = [], []
        for _ in range(self.samples_per_class):
            X.append(self._dog_image().flatten())
            y.append(0)
            X.append(self._cat_image().flatten())
            y.append(1)
        return np.array(X), np.array(y)


class PetClassifier:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel="linear"))
        ])

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class Evaluator:
    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)


def main():
    st.title("Dog vs Cat Image Classifier")

    dataset = SyntheticPetDataset()
    X, y = dataset.load()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    classifier = PetClassifier()
    classifier.train(X_train, y_train)

    predictions = classifier.predict(X_test)
    acc = Evaluator.accuracy(y_test, predictions)

    st.write(f"Model accuracy: **{acc:.2f}**")

    sample_idx = st.slider("View sample image", 0, len(X_test) - 1, 0)
    image = X_test[sample_idx].reshape(32, 32)
    label = "Cat" if predictions[sample_idx] == 1 else "Dog"

    st.image(image, caption=f"Predicted: {label}", clamp=True)


if __name__ == "__main__":
    main()