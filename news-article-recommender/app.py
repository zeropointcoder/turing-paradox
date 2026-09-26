import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class NewsDataset:
    def __init__(self):
        self.articles = [
            "The UK government announced new climate targets focusing on renewable energy.",
            "Manchester United secured a late victory in the Premier League match.",
            "Scientists in Cambridge developed a breakthrough in battery technology.",
            "The Bank of England reviewed interest rates amid inflation concerns.",
            "A new art exhibition opened in London showcasing modern British artists.",
            "Researchers warn about rising sea levels affecting coastal towns in England.",
            "The NHS plans digital upgrades to improve patient care services.",
            "British startups attract record investment in artificial intelligence.",
            "Transport strikes across the UK caused major commuter disruptions.",
            "Parliament debated new education reforms for secondary schools."
        ]


class NewsVectoriser:
    def __init__(self):
        self.vectoriser = TfidfVectorizer(stop_words="english")

    def fit_transform(self, documents):
        return self.vectoriser.fit_transform(documents)

    def transform(self, query):
        return self.vectoriser.transform([query])


class NewsRecommender:
    def __init__(self, dataset, vectoriser):
        self.dataset = dataset
        self.vectoriser = vectoriser
        self.article_vectors = self.vectoriser.fit_transform(self.dataset.articles)

    def recommend(self, query, top_k=3):
        query_vec = self.vectoriser.transform(query)
        scores = cosine_similarity(query_vec, self.article_vectors)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.dataset.articles[i], scores[i]) for i in top_indices]


class StreamlitUI:
    def __init__(self):
        self.dataset = NewsDataset()
        self.vectoriser = NewsVectoriser()
        self.recommender = NewsRecommender(self.dataset, self.vectoriser)

    def run(self):
        st.title("News Article Recommender")
        st.write("Enter a topic or sentence to receive relevant UK news articles.")

        query = st.text_input("Search query")

        if query:
            results = self.recommender.recommend(query)
            st.subheader("Recommended articles")
            for article, score in results:
                st.markdown(f"**Relevance:** {score:.2f}")
                st.write(article)
                st.divider()


if __name__ == "__main__":
    StreamlitUI().run()