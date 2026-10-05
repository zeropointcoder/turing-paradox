import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# Dataset Generator
class SyntheticMedicalDataset:
    def __init__(self, img_size=(64,64), n_samples=100):
        self.img_size = img_size
        self.n_samples = n_samples

    def generate(self):
        normal = np.zeros((self.n_samples, *self.img_size))
        for i in range(self.n_samples):
            rr, cc = np.ogrid[:self.img_size[0], :self.img_size[1]]
            circle = (rr - self.img_size[0]//2)**2 + (cc - self.img_size[1]//2)**2 < (self.img_size[0]//4)**2
            normal[i][circle] = 1.0

        abnormal = np.zeros((self.n_samples, *self.img_size))
        start, end = self.img_size[0]//4, 3*self.img_size[0]//4
        for i in range(self.n_samples):
            abnormal[i][start:end, start:end] = 1.0

        X = np.concatenate([normal, abnormal], axis=0)
        y = np.array([0]*self.n_samples + [1]*self.n_samples)
        return train_test_split(X, y, test_size=0.2, random_state=42)


# Classifier
class MedicalMLP:
    def __init__(self):
        self.clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=42)

    def train(self, X_train, y_train):
        n_samples = len(X_train)
        self.clf.fit(X_train.reshape(n_samples, -1), y_train)

    def evaluate(self, X_test, y_test):
        n_samples = len(X_test)
        y_pred = self.clf.predict(X_test.reshape(n_samples, -1))
        return classification_report(y_test, y_pred)


# Streamlit Interface
st.title("Medical Image Classifier")

# Generate dataset
dataset = SyntheticMedicalDataset()
X_train, X_test, y_train, y_test = dataset.generate()

# Preview a few images
st.subheader("Sample Generated Images")
cols = st.columns(4)
for i, col in enumerate(cols):
    idx = i
    img = X_train[idx]
    label = "Normal" if y_train[idx]==0 else "Abnormal"
    col.image(img, width=100, caption=label)

# Train model
if st.button("Train Model"):
    status = st.empty(); status.info("Training in progress...")
    model = MedicalMLP()
    model.train(X_train, y_train)
    status.success("Training completed!")

    st.subheader("Classification Report")
    report = model.evaluate(X_test, y_test)
    st.text(report)