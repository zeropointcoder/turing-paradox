import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class DatasetBuilder:
    def build(self):
        uk_english = [
            "I prefer colour over color.",
            "The theatre is located in the city centre.",
            "He organised the programme properly.",
            "She travelled by aeroplane.",
            "The cheque was sent yesterday."
        ]

        non_english = [
            "La biblioteca está cerrada hoy.",
            "Das Wetter ist heute sehr schön.",
            "La macchina è parcheggiata fuori.",
            "El niño come una manzana.",
            "Le chat dort sur le canapé."
        ]

        texts = uk_english + non_english
        labels = ["UK English"] * len(uk_english) + ["Non-English"] * len(non_english)
        return texts, labels
    

class LanguageModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", MultinomialNB())
        ])

    def train(self, x, y):
        self.pipeline.fit(x, y)

    def predict(self, text):
        return self.pipeline.predict([text])[0]


class AppController:
    def __init__(self):
        self.dataset = DatasetBuilder()
        self.model = LanguageModel()

    def run(self):
        x, y = self.dataset.build()
        self.model.train(x, y)

        st.title("Language Detection - UK English")
        user_input = st.text_area("Enter text for detection:")

        if st.button("Detect Language") and user_input.strip():
            prediction = self.model.predict(user_input)
            if prediction == "UK English":
                st.success(f"Prediction: {prediction}")
            else:
                st.warning(f"Prediction: {prediction}")


if __name__ == "__main__":
    AppController().run()