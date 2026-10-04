import streamlit as st
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# UK-English synthetic dataset
TEXT_SAMPLES = [
    "London is the capital of England and the United Kingdom.",
    "Barack Obama met with Theresa May in London.",
    "The University of Oxford is one of the oldest universities in the world.",
    "Manchester United played against Liverpool at Old Trafford.",
    "Queen Elizabeth II visited Edinburgh last summer.",
    "The London Eye attracts millions of tourists every year.",
    "Harry Potter was written by J.K. Rowling in Edinburgh.",
    "The British Museum is located in the Bloomsbury area of London.",
    "Liverpool FC won the Premier League title in 2020.",
    "Wimbledon is the oldest tennis tournament in the world.",
    "Oxford University Press publishes academic books worldwide.",
    "Prime Minister Boris Johnson addressed the nation yesterday.",
    "Stonehenge is one of the most famous landmarks in England.",
    "Chelsea FC defeated Arsenal at Stamford Bridge.",
    "The Tower of London houses the Crown Jewels.",
    "Sir David Attenborough has inspired generations of naturalists.",
    "Cambridge University hosts many international students.",
    "Manchester Airport is one of the busiest in the UK.",
    "Emma Watson studied at the University of Oxford.",
    "The Royal Opera House is located in Covent Garden.",
    "Scotland Yard is the headquarters of the Metropolitan Police.",
    "Edinburgh Castle overlooks the city from Castle Rock.",
    "The London Underground is the oldest metro system in the world.",
    "Sir Winston Churchill led Britain during World War II.",
    "The National Gallery displays works by Van Gogh and Turner."
]

LABELS = [
    [("London", "LOCATION"), ("England", "LOCATION"), ("United Kingdom", "LOCATION")],
    [("Barack Obama", "PERSON"), ("Theresa May", "PERSON"), ("London", "LOCATION")],
    [("University of Oxford", "ORG")],
    [("Manchester United", "ORG"), ("Liverpool", "LOCATION"), ("Old Trafford", "LOCATION")],
    [("Queen Elizabeth II", "PERSON"), ("Edinburgh", "LOCATION")],
    [("London Eye", "ORG")],
    [("Harry Potter", "ORG"), ("J.K. Rowling", "PERSON"), ("Edinburgh", "LOCATION")],
    [("British Museum", "ORG"), ("London", "LOCATION"), ("Bloomsbury", "LOCATION")],
    [("Liverpool FC", "ORG")],
    [("Wimbledon", "ORG")],
    [("Oxford University Press", "ORG")],
    [("Boris Johnson", "PERSON")],
    [("Stonehenge", "ORG"), ("England", "LOCATION")],
    [("Chelsea FC", "ORG"), ("Arsenal", "ORG"), ("Stamford Bridge", "LOCATION")],
    [("Tower of London", "ORG"), ("Crown Jewels", "ORG")],
    [("David Attenborough", "PERSON")],
    [("Cambridge University", "ORG")],
    [("Manchester Airport", "ORG"), ("UK", "LOCATION")],
    [("Emma Watson", "PERSON"), ("University of Oxford", "ORG")],
    [("Royal Opera House", "ORG"), ("Covent Garden", "LOCATION")],
    [("Scotland Yard", "ORG"), ("Metropolitan Police", "ORG")],
    [("Edinburgh Castle", "ORG"), ("Castle Rock", "LOCATION")],
    [("London Underground", "ORG")],
    [("Winston Churchill", "PERSON"), ("Britain", "LOCATION"), ("World War II", "ORG")],
    [("National Gallery", "ORG"), ("Van Gogh", "PERSON"), ("Turner", "PERSON")]
]


class NERModel:
    def __init__(self):
        self.vectoriser = TfidfVectorizer()
        self.models = {}  # one classifier per entity type
        self.entity_types = ["PERSON", "LOCATION", "ORG"]
        self.corpus = TEXT_SAMPLES
        self.labels = LABELS
        self._train_models()

    def _train_models(self):
        X = self.vectoriser.fit_transform(self.corpus)
        for ent_type in self.entity_types:
            y = []
            for sent_labels in self.labels:
                # if entity type exists in sentence → 1 else 0
                y.append(int(any(label == ent_type for _, label in sent_labels)))
            clf = LogisticRegression()
            clf.fit(X, y)
            self.models[ent_type] = clf

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        entities_found = []
        X_input = self.vectoriser.transform([text])
        for ent_type, clf in self.models.items():
            if clf.predict(X_input)[0] == 1:
                # simple rule: find words that exist in synthetic dictionary
                for sent_labels in self.labels:
                    for word, label in sent_labels:
                        if label == ent_type and word in text:
                            entities_found.append((word, label))
        return entities_found


class NERApp:
    def __init__(self):
        self.model = NERModel()

    def run(self):
        st.title("NER Text Explorer (ML Upgrade)")
        st.markdown(
            "Extracts **Person**, **Organisation**, and **Location** entities from UK English text using offline ML models."
        )
        choice = st.selectbox("Choose text input:", ["Sample text", "Custom text"])
        if choice == "Sample text":
            sample_text = st.selectbox("Select a sample sentence:", TEXT_SAMPLES)
            text = sample_text
        else:
            text = st.text_area("Enter your text here:", height=150)

        if st.button("Extract Entities"):
            entities = self.model.extract_entities(text)
            if entities:
                st.subheader("Entities Found:")
                for ent, label in entities:
                    st.write(f"{ent} → {label}")
            else:
                st.info("No entities found.")


if __name__ == "__main__":
    app = NERApp()
    app.run()