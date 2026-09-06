import streamlit as st
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


class DigitDataset:
    def load(self):
        data = load_digits()
        X = data.images.reshape(len(data.images), -1)
        y = data.target
        return train_test_split(X, y, test_size=0.2, random_state=42)


class DigitModel:
    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            max_iter=300,
            random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class Evaluator:
    def evaluate(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        return accuracy_score(y_test, predictions)


class DigitApp:
    def __init__(self):
        self.dataset = DigitDataset()
        self.model = DigitModel()
        self.evaluator = Evaluator()

    def run(self):
        st.title("Handwritten Digit Recognition")

        X_train, X_test, y_train, y_test = self.dataset.load()
        self.model.train(X_train, y_train)

        accuracy = self.evaluator.evaluate(self.model, X_test, y_test)
        st.write(f"Model accuracy: **{accuracy:.2%}**")

        index = st.slider("Select a test image", 0, len(X_test) - 1, 0)

        image = X_test[index].reshape(8, 8) / 16.0
        prediction = self.model.predict([X_test[index]])[0]

        st.image(
            image,
            width=150,
            caption=f"Predicted digit: {prediction}"
        )


if __name__ == "__main__":
    DigitApp().run()