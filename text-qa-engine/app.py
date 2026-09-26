import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeCorpus:
    def __init__(self):
        self.documents = [
            "Machine learning is a field of artificial intelligence that enables systems to learn from data.",
            "Supervised learning uses labelled data to train predictive models.",
            "Unsupervised learning finds patterns in data without labelled outputs.",
            "Deep learning uses neural networks with multiple layers to model complex patterns.",
            "Natural language processing focuses on the interaction between computers and human language.",
            "Overfitting occurs when a model learns noise instead of general patterns.",
            "Regularisation reduces model complexity to improve generalisation.",
            "Classification predicts discrete class labels.",
            "Regression predicts continuous numerical values.",
            "Evaluation metrics measure model performance such as accuracy and precision."
        ]


class VectorModel:
    def __init__(self):
        self.vectoriser = TfidfVectorizer(stop_words="english")
        self.matrix = None

    def fit(self, texts):
        self.matrix = self.vectoriser.fit_transform(texts)

    def transform(self, query):
        return self.vectoriser.transform([query])


class QuestionAnsweringSystem:
    def __init__(self, corpus):
        self.corpus = corpus
        self.model = VectorModel()
        self.model.fit(self.corpus.documents)

    def answer(self, question):
        query_vec = self.model.transform(question)
        scores = cosine_similarity(query_vec, self.model.matrix)
        best_index = int(np.argmax(scores))
        return self.corpus.documents[best_index], float(scores[0, best_index])


class Evaluator:
    @staticmethod
    def confidence_label(score):
        if score > 0.4:
            return "High confidence"
        if score > 0.2:
            return "Medium confidence"
        return "Low confidence"


st.set_page_config(page_title="Question Answering System", layout="centered")
st.title("Question Answering System")

qa_system = QuestionAnsweringSystem(KnowledgeCorpus())

user_question = st.text_input("Enter your question")

if user_question:
    answer, score = qa_system.answer(user_question)
    confidence = Evaluator.confidence_label(score)

    st.subheader("Answer")
    st.write(answer)

    st.caption(f"{confidence} (similarity score: {score:.2f})")