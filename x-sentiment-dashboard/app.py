import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TweetDataset:
    def __init__(self):
        self.data = self._build_dataset()

    def _build_dataset(self):
        tweets = [
            ("Absolutely love the new train schedule, much better service", "positive"),
            ("This weather is dreadful, trains delayed again", "negative"),
            ("Had an average experience at the station today", "neutral"),
            ("Brilliant support staff, very helpful and polite", "positive"),
            ("Ticket prices are ridiculous, not impressed at all", "negative"),
            ("Journey was fine, nothing special really", "neutral"),
            ("Really pleased with how smooth the commute was", "positive"),
            ("Terrible communication during delays, very frustrating", "negative"),
            ("Everything worked as expected", "neutral"),
        ]
        return pd.DataFrame(tweets, columns=["text", "sentiment"])


class SentimentModel:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english")),
                ("clf", LogisticRegression(max_iter=1000)),
            ]
        )

    def train(self, x, y):
        self.pipeline.fit(x, y)

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)


class SentimentDashboard:
    def __init__(self):
        st.set_page_config(page_title="Twitter Sentiment Dashboard", layout="wide")
        self.dataset = TweetDataset()
        self.model = SentimentModel()
        self.model.train(
            self.dataset.data["text"], self.dataset.data["sentiment"]
        )

    def run(self):
        st.title("Twitter Sentiment Dashboard")

        st.subheader("Sample dataset")
        st.dataframe(self.dataset.data, width=900)

        st.subheader("Analyse a tweet")
        user_input = st.text_area(
            "Enter a tweet-style message (UK English)",
            height=100,
        )

        if user_input.strip():
            prediction = self.model.predict([user_input])[0]
            probabilities = self.model.predict_proba([user_input])[0]

            result_df = pd.DataFrame(
                {
                    "Sentiment": self.model.pipeline.classes_,
                    "Confidence": np.round(probabilities, 3),
                }
            )

            st.markdown(f"### Predicted Sentiment: **{prediction.capitalize()}**")
            st.bar_chart(
                result_df.set_index("Sentiment"),
                width=600,
            )


if __name__ == "__main__":
    SentimentDashboard().run()