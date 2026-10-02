import re
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st


class ReviewCleaner:
    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z\s\.]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"\.\s+", text)
        return [s.strip() for s in sentences if len(s.split()) > 4]


class TfidfSummariser:
    def __init__(self):
        self.cleaner = ReviewCleaner()
        self.vectoriser = TfidfVectorizer(stop_words="english")

    def summarise(self, text: str, top_k: int = 3) -> List[str]:
        clean_text = self.cleaner.clean(text)
        sentences = self.cleaner.split_sentences(clean_text)
        if not sentences:
            return []

        tfidf_matrix = self.vectoriser.fit_transform(sentences)
        # Compute mean vector and convert to 2D numpy array
        corpus_mean = np.asarray(tfidf_matrix.mean(axis=0))
        corpus_mean = corpus_mean.reshape(1, -1)  # shape (1, n_features)

        scores = cosine_similarity(tfidf_matrix, corpus_mean).flatten()
        ranked_idx = np.argsort(scores)[::-1]
        return [sentences[i].capitalize() for i in ranked_idx[:top_k]]


class ReviewApp:
    def run(self):
        st.title("Product Review Summariser (ML TF-IDF)")
        st.write("Paste multiple product reviews below to generate a concise summary.")

        reviews = st.text_area("Product reviews", height=220)
        sentence_count = st.slider("Summary length (sentences)", 1, 5, 3)

        if st.button("Summarise"):
            if not reviews.strip():
                st.warning("Please enter product reviews.")
                return

            summariser = TfidfSummariser()
            summary = summariser.summarise(reviews, sentence_count)

            st.subheader("Summary")
            for s in summary:
                st.markdown(f"- {s}.")


if __name__ == "__main__":
    ReviewApp().run()