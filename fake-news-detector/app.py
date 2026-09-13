import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


class NewsDataset:
    def load(self):
        texts = [
            "The government announced a new policy to improve public transport across the country",
            "Scientists confirm climate change is accelerating faster than expected",
            "The football club secured a historic victory after extra time",
            "Doctors recommend regular exercise to maintain good health",
            "The prime minister addressed parliament regarding economic reforms",
            "Aliens landed in London and demanded unlimited tea supplies",
            "A secret device can instantly change your hair colour forever",
            "Politicians replaced by robots according to leaked documents",
            "Eating one apple grants immortality scientists refuse to explain",
            "Time travellers spotted correcting history in Manchester"
        ]

        labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        return pd.DataFrame({"text": texts, "label": labels})
    

class FakeNewsModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfid", TfidfVectorizer(
                stop_words="english", 
                ngram_range=(1, 2), 
                max_features=3000)
            ),
            ("clf", LogisticRegression(max_iter=1000))
        ])

    def train(self, x, y):
        self.pipeline.fit(x, y)

    def predict(self, text):
        return self.pipeline.predict([text])[0]

    def evaluate(self, x, y):
        preds = self.pipeline.predict(x)
        return accuracy_score(y, preds)


class App:
    def __init__(self):
        self.dataset = NewsDataset()
        self.model = FakeNewsModel()

    def run(self):
        st.title("Fake News Detector")
        data = self.dataset.load()
        self.model.train(data["text"], data["label"])
        accuracy = self.model.evaluate(data["text"], data["label"])

        st.write(f"Model accuracy on offline sample data: **{accuracy: .2f}**")

        user_input = st.text_area(
            "Enter a news headline or short article:",
            height=150
        )

        if st.button("Analyse"):
            if user_input.strip():
                prediction = self.model.predict(user_input)
                if prediction == 1:
                    st.error("⚠️ Likely Fake News")
                else:
                    st.success("✅ Likely Real News")
            else:
                st.warning("Please enter some text.")
                

if __name__ == "__main__":
    App().run()