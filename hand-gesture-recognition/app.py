import streamlit as st
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class GestureDataset:
    def __init__(self, samples_per_class=200):
        self.samples_per_class = samples_per_class
        self.gestures = ["Open Hand", "Fist", "Peace"]
        self.features = 21 * 2

    def _generate_gesture(self, gesture_id):
        base = np.random.rand(self.samples_per_class, self.features)
        noise = np.random.normal(0, 0.03, base.shape)

        if gesture_id == 0:
            pattern = np.linspace(0.2, 0.9, self.features)
        elif gesture_id == 1: 
            pattern = np.linspace(0.1, 0.4, self.features)
        else:
            pattern = np.sin(np.linspace(0, 3, self.features))

        return base + pattern + noise

    def load(self):
        X, y = [], []
        for idx in range(len(self.gestures)):
            data = self._generate_gesture(idx)
            X.append(data)
            y.extend([idx] * len(data))

        return np.vstack(X), np.array(y), self.gestures


class GestureModel:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=5)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class Evaluator:
    def evaluate(self, model, X, y):
        preds = model.predict(X)
        return accuracy_score(y, preds)


class GestureApp:
    def __init__(self):
        self.dataset = GestureDataset()
        self.model = GestureModel()
        self.evaluator = Evaluator()

    def _init_state(self):
        if "sample" not in st.session_state:
            st.session_state.sample = None
            st.session_state.actual = None
            st.session_state.predicted = None

    def run(self):
        st.title("Hand Gesture Recognition")

        self._init_state()

        X, y, labels = self.dataset.load()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        self.model.train(X_train, y_train)
        accuracy = self.evaluator.evaluate(self.model, X_test, y_test)

        st.metric("Model accuracy", f"{accuracy:.2f}")

        st.subheader("Try a synthetic hand sample")

        if st.button("Generate new sample"):
            gesture_id = np.random.randint(0, len(labels))
            gesture_samples = X[y == gesture_id]
            sample = gesture_samples[np.random.randint(len(gesture_samples))].reshape(1, -1)

            prediction = self.model.predict(sample)[0]

            st.session_state.sample = sample
            st.session_state.actual = labels[gesture_id]
            st.session_state.predicted = labels[prediction]

        if st.session_state.sample is not None:
            st.write("Actual gesture:", st.session_state.actual)
            st.write("Predicted gesture:", st.session_state.predicted)


if __name__ == "__main__":
    GestureApp().run()