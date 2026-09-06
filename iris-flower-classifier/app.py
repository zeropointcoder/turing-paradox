import numpy as np
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


class IrisData:
    def __init__(self):
        dataset = load_iris()
        self.X = dataset.data
        self.y = dataset.target
        self.feature_names = dataset.feature_names
        self.target_names = dataset.target_names

    def split(self, test_size=0.2, random_state=42):
        return train_test_split(
            self.X, 
            self.y, 
            test_size=test_size, 
            random_state=random_state
        )


class IrisModel:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("scalar", StandardScaler()),
                ("model", LogisticRegression(max_iter=200))
            ]
        )

    def train(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


class Trainer:
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def run(self):
        X_train, X_test, y_train, y_test = self.data.split()
        self.model.train(X_train, y_train)
        return X_test, y_test

class Evaluator:
    def __init__(self, model):
        self.model = model

    def accuracy(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        return accuracy_score(y_test, predictions)

        
class AppUI:
    def __init__(self):
        st.set_page_config(page_title="Iris Flower Classification", layout="centered")
        self.data = IrisData()
        self.model = IrisModel()

    def show_example_inputs(self):
        st.markdown(
            """
# Example measurements (use these to test)

You can copy the following **exact measurements** into the sliders to verify predictions.
These values come from real samples in the Iris dataset.

**Iris Setosa**
- Sepal length: **5.1**
- Sepal width: **3.5**
- Petal length: **1.4**
- Petal width: **0.2**

**Iris Versicolour**
- Sepal length: **7.0**
- Sepal width: **3.2**
- Petal length: **4.7**
- Petal width: **1.4**

**Iris Virginica**
- Sepal length: **6.3**
- Sepal width: **3.3**
- Petal length: **6.0**
- Petal width: **2.5**
"""
        )

    def run(self):
        st.title("Iris Flower Classification")
        st.write("Predict the species using flower measurements.")

        self.show_example_inputs()

        trainer = Trainer(self.model, self.data)
        X_test, y_test = trainer.run()

        evaluator = Evaluator(self.model)
        accuracy = evaluator.accuracy(X_test, y_test)

        st.subheader("Model performance")
        st.write(f"Accuracy: **{accuracy:.2f}**")

        st.subheader("Enter flower measurements")

        inputs = []
        for name in self.data.feature_names:
            value = st.slider(name, 0.0, 8.0, 4.0)
            inputs.append(value)

        input_array = np.array(inputs).reshape(1, -1)

        if st.button("Predict"):
            prediction = self.model.predict(input_array)[0]
            probabilities = self.model.predict_proba(input_array)[0]

            st.success(
                f"Predicted species: **{self.data.target_names[prediction]}**"
            )

            st.write("Prediction probabilities:")
            for label, prob in zip(self.data.target_names, probabilities):
                st.write(f"- {label}: {prob:.2f}")


if __name__ == "__main__":
    AppUI().run()