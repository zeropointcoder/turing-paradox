import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import re


class TranslatorML:
    def __init__(self):
        # Original dictionary for initial training
        self.uk_vocab = {
            "color": "colour",
            "favorite": "favourite",
            "organize": "organise",
            "analyze": "analyse",
            "theater": "theatre",
            "center": "centre",
            "apologize": "apologise",
            "catalog": "catalogue",
            "traveler": "traveller",
            "defense": "defence",
            "license": "licence",
            "program": "programme",
            "jewelry": "jewellery",
            "check": "cheque",
            "dialog": "dialogue",
            "gray": "grey",
            "mom": "mum",
            "offense": "offence",
            "plow": "plough",
            "curb": "kerb",
            "maneuver": "manoeuvre",
            "pajama": "pyjama",
            "curbside": "kerbside",
            "enroll": "enrol",
            "installment": "instalment"
        }

        self.model = None
        self.vectoriser = None
        self.train_model()

    def train_model(self):
        us_words = list(self.uk_vocab.keys())
        uk_words = list(self.uk_vocab.values())
        # Character n-gram vectorization
        self.vectoriser = CountVectorizer(analyzer='char', ngram_range=(2,3))
        X = self.vectoriser.fit_transform(us_words)
        self.model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)
        self.model.fit(X, uk_words)

    def translate_word(self, word: str) -> str:
        match = re.match(r"(\w+)(\W*)", word)
        if match:
            core, punct = match.groups()
            X = self.vectoriser.transform([core.lower()])
            predicted = self.model.predict(X)[0]
            if core.istitle():
                predicted = predicted.capitalize()
            return predicted + punct
        return word
    
    def translate_text(self, text: str) -> str:
        words = text.split()
        translated = [self.translate_word(w) for w in words]
        return ' '.join(translated)


class StreamlitUI:
    def __init__(self):
        self.translator = TranslatorML()
        self.setup_ui()

    def setup_ui(self):
        st.title("US → UK English Translator (ML)")
        st.write("Convert US English text to UK English using an ML model.")
        input_text = st.text_area("Enter English text:")
        if st.button("Translate"):
            if input_text.strip() == "":
                st.warning("Please enter some text to translate.")
            else:
                translated = self.translator.translate_text(input_text)
                st.success(translated)


if __name__ == "__main__":
    StreamlitUI()