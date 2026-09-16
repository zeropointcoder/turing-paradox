import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class FaceDatasetGenerator:
    def __init__(self, image_size=32):
        self.image_size = image_size
        self.labels = ["happy", "sad", "angry"]

    def _draw_face(self, expression):
        img = Image.new("L", (self.image_size, self.image_size), color=255)
        d = ImageDraw.Draw(img)

        d.ellipse((6, 6, 26, 26), outline=0, width=2)
        d.ellipse((11, 12, 13, 14), fill=0)
        d.ellipse((19, 12, 21, 14), fill=0)

        if expression == "happy":
            d.arc((12, 14, 20, 22), start=0, end=180, fill=0, width=2)
        elif expression == "sad":
            d.arc((12, 18, 20, 26), start=180, end=360, fill=0, width=2)
        else:
            d.line((12, 20, 20, 20), fill=0, width=2)

        return np.array(img)

    def generate(self, samples_per_class=80):
        X, y = [], []
        for idx, label in enumerate(self.labels):
            for _ in range(samples_per_class):
                img = self._draw_face(label)
                noise = np.random.normal(0, 8, img.shape)
                X.append((img + noise).clip(0, 255).flatten())
                y.append(idx)
        return np.array(X), np.array(y), self.labels


class ExpressionModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ("pca", PCA(n_components=40)),
            ("svm", SVC(kernel="rbf"))
        ])

    def train(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, X):
        return self.pipeline.predict(X)


class Evaluator:
    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)


@st.cache_resource
def train_model():
    generator = FaceDatasetGenerator()
    X, y, labels = generator.generate()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    model = ExpressionModel()
    model.train(X_train, y_train)
    preds = model.predict(X_test)
    acc = Evaluator.accuracy(y_test, preds)
    return model, X_test, y_test, labels, acc


st.set_page_config(page_title="Facial Expression Detector")
st.title("Facial Expression Detector")

model, X_test, y_test, label_names, acc = train_model()

st.metric("Model accuracy", f"{acc:.2%}")

if "sample_idx" not in st.session_state:
    st.session_state.sample_idx = np.random.randint(0, len(X_test))

if st.button("Generate another face"):
    st.session_state.sample_idx = np.random.randint(0, len(X_test))

idx = st.session_state.sample_idx
sample = X_test[idx].reshape(32, 32)
prediction = label_names[model.predict(X_test[idx:idx + 1])[0]]

st.subheader("Predicted expression")
st.image(sample, clamp=True, caption=f"{prediction}")