import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class ReviewDataset:
    def __init__(self):
        self.data = [
            ("An outstanding film with brilliant performances", 1),
            ("Absolutely dreadful and painfully slow", 0),
            ("A wonderfully written and emotionally rich story", 1),
            ("Poorly acted and badly directed", 0),
            ("An enjoyable and heart-warming experience", 1),
            ("Utterly disappointing and not worth the time", 0),
            ("A masterful piece of cinema", 1),
            ("The plot was weak and unconvincing", 0),
            ("Excellent direction and strong acting", 1),
            ("Boring, predictable, and far too long", 0),
            ("A touching and beautifully filmed story", 1),
            ("Messy script and terrible pacing", 0),
        ]

    def load(self):
        texts, labels = zip(*self.data)
        return list(texts), np.array(labels)

class SentimentModel:
    def __init__(self, seed: int = 42):
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
                (
                    "clf", 
                    LogisticRegression(
                        random_state=seed,
                        class_weight="balanced",
                        max_iter=1000,
                    )
                )
            ]
        )

    def train(self, X, y):
        self.pipeline.fit(X, y)

    def predict(self, text: str):
        return self.pipeline.predict([text])[0]


class Evaluator:
    def evaluate(self, model, X_test, y_test):
        preds = model.pipeline.predict(X_test)
        return accuracy_score(y_test, preds)


def main():
    st.set_page_config(page_title="Movie Sentiment Analysis", layout="centered")
    st.title("Movie Review Sentiment Analysis")

    dataset = ReviewDataset()
    X, y = dataset.load()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    model = SentimentModel()
    model.train(X_train, y_train)

    evaluator = Evaluator()
    accuracy = evaluator.evaluate(model, X_test, y_test)

    st.markdown(f"**Model accuracy:** {accuracy:.2f}")

    review = st.text_area("Enter a movie review:")

    if st.button("Analyse sentiment"):
        if review.strip():
            prediction = model.predict(review)
            label = "Positive 😊" if prediction == 1 else "Negative ☹️"
            st.subheader(label)
        else:
            st.warning("Please enter a review.")


if __name__ == "__main__":
    main()