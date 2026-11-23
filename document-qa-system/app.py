import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentStore:
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.vectoriser = TfidfVectorizer(stop_words="english")
        self.embeddings = self.vectoriser.fit_transform(documents)

    def retrieve(self, query: str, top_k: int = 3) -> list[str]:
        query_vec = self.vectoriser.transform([query])
        scores = cosine_similarity(query_vec, self.embeddings).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_indices]


class AnswerEngine:
    def generate(self, query: str, contexts: list[str]) -> str:
        if not contexts:
            return "No relevant information found."
        return contexts[0]


class DocumentQASystem:
    def __init__(self, documents: list[str]):
        self.store = DocumentStore(documents)
        self.engine = AnswerEngine()

    def answer(self, query: str) -> tuple[str, list[str]]:
        contexts = self.store.retrieve(query)
        answer = self.engine.generate(query, contexts)
        return answer, contexts


DOCUMENTS = [
    "Machine learning is a field of artificial intelligence that focuses on pattern recognition.",
    "Retrieval-augmented generation combines information retrieval with text generation.",
    "TF-IDF measures how important a word is to a document in a collection.",
    "Cosine similarity measures the angle between two vectors to estimate similarity.",
    "Streamlit allows rapid development of data applications using Python."
]


st.set_page_config(page_title="Document Q&A", layout="centered")
st.title("Document Q&A System")

qa_system = DocumentQASystem(DOCUMENTS)

query = st.text_input("Ask a question about the documents")

if query:
    answer, sources = qa_system.answer(query)
    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved Contexts"):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**{i}.** {src}")
