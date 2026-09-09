import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class EmailDataset:
    def load(self):
        spam = [
            "Win a cash prize now",
            "Free holiday offer just for you",
            "Urgent account verification required",
            "Claim your reward today",
            "Limited offer act now",
            "You have won a voucher",
            "Exclusive deal waiting for you",
        ]

        ham = [
            "Please find the meeting agenda attached",
            "Can we reschedule our appointment",
            "Invoice for last month's services",
            "Looking forward to our discussion",
            "Here is the project update",
            "Thank you for your email",
            "Let us confirm the delivery date",
        ]

        texts = spam * 10 + ham * 10
        labels = [1] * len(spam) * 10 + [0] * len(ham) * 10

        return pd.DataFrame({"text": texts, "label": labels})


class SpamModel:
    def build(self):
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english")),
                ("model", MultinomialNB())
            ]
        )


class Trainer:
    def train(self, model, X_train, y_train):
        model.fit(X_train, y_train)
        return model


class Evaluator:
    def evaluate(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        return accuracy_score(y_test, predictions)


st.set_page_config(page_title="Spam Email Classifier", layout="centered")
st.title("Spam Email Classifier")

dataset = EmailDataset().load()
X_train, X_test, y_train, y_test = train_test_split(
    dataset["text"], dataset["label"], test_size=0.25, random_state=42
)

model = SpamModel().build()
trainer = Trainer()
model = trainer.train(model, X_train, y_train)

accuracy = Evaluator().evaluate(model, X_test, y_test)
st.caption(f"Model accuracy: {accuracy:.2f}")

email_input = st.text_area("Paste an email message below:")

if st.button("Classify"):
    if email_input.strip():
        prediction = model.predict([email_input])[0]
        label = "Spam" if prediction == 1 else "✅ Legitimate"
        st.subheader(label)