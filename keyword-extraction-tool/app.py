import streamlit as st
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import random
import numpy as np


class DatasetGenerator:
    def __init__(self, n_samples_per_category=12):
        self.n_samples = n_samples_per_category
        self.categories = ['Sports', 'Tech', 'Politics']
        self.samples = self._generate_dataset()

    def _generate_sentence(self, category):
        if category == 'Sports':
            subjects = ['football match', 'basketball game', 'tennis championship', 'cricket match', 'rugby final', 'marathon race']
            verbs = ['was thrilling', 'ended in a tie', 'starts next week', 'won by a large margin', 'attracted many fans', 'set a new record']
        elif category == 'Tech':
            subjects = ['AI technology', 'machine learning', 'quantum computing', 'new smartphone', 'software update', 'cybersecurity system']
            verbs = ['is emerging rapidly', 'improves every year', 'shows remarkable progress', 'features enhanced performance', 'was released today', 'requires constant monitoring']
        else:  # Politics
            subjects = ['government', 'parliament', 'mayor', 'senators', 'opposition', 'council']
            verbs = ['passed a new law', 'debated policies', 'announced new plans', 'criticised proposals', 'approved reforms', 'voted on changes']

        subject = random.choice(subjects)
        verb = random.choice(verbs)
        return f"The {subject} {verb}"

    def _generate_dataset(self):
        texts = []
        categories = []
        for cat in self.categories:
            for _ in range(self.n_samples):
                texts.append(self._generate_sentence(cat))
                categories.append(cat)
        return pd.DataFrame({'text': texts, 'category': categories})


class KeywordExtractor:
    def __init__(self, max_keywords=10):
        self.max_keywords = max_keywords
        self.vectoriser = TfidfVectorizer(stop_words='english', ngram_range=(1,2))

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text

    def extract_keywords_with_weights(self, text):
        text_clean = self.clean_text(text)
        if not text_clean.strip():
            return [("No valid words found", 0)]

        tfidf_matrix = self.vectoriser.fit_transform([text_clean])
        feature_array = self.vectoriser.get_feature_names_out()
        tfidf_values = tfidf_matrix.toarray().flatten()
        indices = tfidf_values.argsort()[::-1][:self.max_keywords]
        return [(feature_array[i], round(tfidf_values[i], 3)) for i in indices]


class TextClassifier:
    def __init__(self, dataset: pd.DataFrame):
        self.model = LogisticRegression()
        self.vectoriser = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
        self.encoder = LabelEncoder()
        self._train_offline_model(dataset)

    def _train_offline_model(self, data):
        X = self.vectoriser.fit_transform(data['text'])
        y = self.encoder.fit_transform(data['category'])
        self.model.fit(X, y)

    def predict_category(self, text):
        X = self.vectoriser.transform([text])
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X).max()  # confidence
        return self.encoder.inverse_transform(y_pred)[0], round(y_prob, 3)


class App:
    def __init__(self):
        st.set_page_config(page_title="Keyword + Text Classifier", layout="wide")
        st.title("Keyword Extraction & Text Classification Tool")
        dataset = DatasetGenerator().samples
        self.extractor = KeywordExtractor()
        self.classifier = TextClassifier(dataset)

    def run(self):
        st.markdown("Enter text below to extract keywords and predict its category with confidence:")
        user_text = st.text_area("Input Text", height=200)
        max_kw = st.slider("Max Keywords", min_value=5, max_value=20, value=10)
        self.extractor.max_keywords = max_kw

        if st.button("Process Text"):
            if user_text.strip():
                keywords = self.extractor.extract_keywords_with_weights(user_text)
                category, confidence = self.classifier.predict_category(user_text)

                st.markdown("### Extracted Keywords (with weights)")
                st.write(keywords)

                st.markdown("### Predicted Category")
                st.write(f"{category} (Confidence: {confidence})")
            else:
                st.warning("Please enter some text.")


if __name__ == "__main__":
    App().run()
