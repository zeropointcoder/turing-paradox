import streamlit as st
import re
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier


# ML Model
class ResumeMLModel:
    """Offline ML model for resume scoring"""
    def __init__(self):
        self.data = pd.DataFrame({
            "resume": [
                "Developed software, managed team, implemented features",
                "Worked on tasks, did stuff, handled things",
                "Led project, designed system, optimised processes",
                "Assisted colleagues, completed jobs, did activities"
            ],
            "label": [1, 0, 1, 0]  # 1 = strong action verbs, 0 = weak
        })
        self.vectoriser = TfidfVectorizer()
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self._train_model()

    def _train_model(self):
        X = self.vectoriser.fit_transform(self.data["resume"])
        y = self.data["label"]
        self.model.fit(X, y)

    def predict_score(self, text: str) -> float:
        X_new = self.vectoriser.transform([text])
        prob = self.model.predict_proba(X_new)[0][1]
        return round(prob * 100, 2)


# Resume Analyser
class ResumeAnalyser:
    def __init__(self):
        self.uk_words = set([
            "colour", "favour", "optimise", "organisation",
            "centre", "analyse", "modelling", "programme",
            "behaviour", "labour", "defence", "travelling",
            "licence", "practise"
        ])
        self.us_to_uk = {
            "color": "colour", "analyze": "analyse", "optimize": "optimise",
            "organize": "organise", "center": "centre", "modeling": "modelling",
            "program": "programme", "behavior": "behaviour", "license": "licence"
        }
        # Common English stopwords to ignore in keyword density
        self.stopwords = set([
            "a", "an", "the", "and", "or", "for", "on", "in", "of", "to", "with", "at", "by", "from", "as", "is", "was", "were", "be", "this", "that", "it"
        ])

    def clean_text(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z']+", text.lower())

    def spelling_issues(self, words: list[str]) -> list[str]:
        # Only flag words not in UK list and longer than 2 letters
        return sorted([w for w in words if len(w) > 2 and w not in self.uk_words and w not in self.stopwords])

    def us_spelling_warnings(self, words: list[str]) -> list[str]:
        return sorted({f"{w} → {self.us_to_uk[w]}" for w in words if w in self.us_to_uk})

    def keyword_density(self, words: list[str]) -> list[tuple]:
        # Filter out stopwords
        filtered = [w for w in words if w not in self.stopwords]
        return Counter(filtered).most_common(8)


# Resume Evaluator
class ResumeEvaluator:
    def __init__(self, analyser: ResumeAnalyser, ml_model: ResumeMLModel):
        self.analyser = analyser
        self.ml_model = ml_model

    def evaluate(self, text: str) -> dict:
        words = self.analyser.clean_text(text)
        return {
            "word_count": len(words),
            "spelling": self.analyser.spelling_issues(words),
            "us_spellings": self.analyser.us_spelling_warnings(words),
            "keywords": self.analyser.keyword_density(words),
            "ml_score": self.ml_model.predict_score(text)
        }


# Streamlit UI
class ResumeUI:
    def __init__(self):
        self.analyser = ResumeAnalyser()
        self.ml_model = ResumeMLModel()
        self.evaluator = ResumeEvaluator(self.analyser, self.ml_model)

    def run(self):
        st.set_page_config(page_title="ML Resume Analyser", layout="centered")
        st.title("Offline ML Resume Analyser")

        resume_text = st.text_area("Paste your resume text below", height=320)

        if st.button("Analyse Resume"):
            if not resume_text.strip():
                st.warning("Please enter resume text to analyse.")
                return

            results = self.evaluator.evaluate(resume_text)

            st.subheader("Summary")
            st.write(f"Word count: **{results['word_count']}**")
            st.write(f"ML-based action verb score: **{results['ml_score']}%**")

            st.subheader("Spelling issues")
            if results["spelling"]:
                st.write(", ".join(results["spelling"]))
            else:
                st.success("No spelling issues detected.")

            if results["us_spellings"]:
                st.subheader("UK spelling suggestions")
                for item in results["us_spellings"]:
                    st.write(item)

            st.subheader("Top keyword frequency")
            for word, freq in results["keywords"]:
                st.write(f"{word}: {freq}")


if __name__ == "__main__":
    ResumeUI().run()