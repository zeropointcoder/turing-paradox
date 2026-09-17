import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ResumeDataset:
    def __init__(self):
        self.data = [
            {
                "name": "Alice Brown",
                "text": "Data analyst with experience in Python, SQL, statistics, and business reporting."
            },
            {
                "name": "James Wilson",
                "text": "Software engineer skilled in Java, system design, cloud platforms, and backend development."
            },
            {
                "name": "Sophie Patel",
                "text": "Machine learning engineer experienced in Python, scikit learn, data modelling, and AI systems."
            },
            {
                "name": "Oliver Smith",
                "text": "Business analyst specialising in stakeholder management, dashboards, Excel, and data insights."
            }
        ]

    def to_dataframe(self):
        return pd.DataFrame(self.data)


class ResumeVectoriser:
    def __init__(self):
        self.vectoriser = TfidfVectorizer(stop_words="english")

    def fit_transform(self, documents):
        return self.vectoriser.fit_transform(documents)

    def transform(self, document):
        return self.vectoriser.transform([document])


class ResumeScorer:
    @staticmethod
    def score(resume_vectors, job_vector):
        similarities = cosine_similarity(resume_vectors, job_vector)
        return similarities.flatten()


class ResumeScreeningPipeline:
    def __init__(self):
        self.dataset = ResumeDataset()
        self.vectoriser = ResumeVectoriser()

    def run(self, job_description):
        df = self.dataset.to_dataframe()
        resume_vectors = self.vectoriser.fit_transform(df["text"])
        job_vector = self.vectoriser.transform(job_description)
        scores = ResumeScorer.score(resume_vectors, job_vector)
        df["match_score"] = np.round(scores, 3)
        return df.sort_values("match_score", ascending=False)


def main():
    st.set_page_config(page_title="Resume Screening Model", layout="centered")
    st.title("Resume Screening Model")

    st.write("Enter a job description to rank candidate CVs by relevance.")

    job_description = st.text_area(
        "Job Description",
        height=150,
        placeholder="Example: Python developer with experience in machine learning and data analysis."
    )

    if st.button("Screen Resumes") and job_description.strip():
        pipeline = ResumeScreeningPipeline()
        results = pipeline.run(job_description)
        st.subheader("Ranked Candidates")
        st.dataframe(results, width="stretch")


if __name__ == "__main__":
    main()