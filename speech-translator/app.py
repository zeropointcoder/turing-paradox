import streamlit as st
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


class LanguageDetectorML:
    def __init__(self):
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(1,2))
        self.model = MultinomialNB()
        # Expanded offline training data
        self.sentences = [
            # French
            ("bonjour je suis content merci", "French"),
            ("je m'appelle pierre et j'aime le chocolat", "French"),
            ("il fait beau aujourd'hui", "French"),
            ("merci beaucoup pour votre aide", "French"),
            ("je vais à l'école tous les jours", "French"),
            # Spanish
            ("hola estoy feliz gracias", "Spanish"),
            ("me llamo maria y me gusta el chocolate", "Spanish"),
            ("hace sol hoy", "Spanish"),
            ("muchas gracias por tu ayuda", "Spanish"),
            ("voy a la escuela todos los días", "Spanish"),
            # German
            ("hallo ich bin froh danke", "German"),
            ("mein name ist hans und ich liebe schokolade", "German"),
            ("heute ist es sonnig", "German"),
            ("vielen dank für deine hilfe", "German"),
            ("ich gehe jeden tag zur schule", "German"),
            # English
            ("hello i am happy thank you", "English"),
            ("my name is john and i like chocolate", "English"),
            ("it is sunny today", "English"),
            ("thank you very much for your help", "English"),
            ("i go to school every day", "English")
        ]
        self._train_model()

    def _train_model(self):
        texts, labels = zip(*self.sentences)
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)

    def predict(self, text):
        X_test = self.vectorizer.transform([text])
        return self.model.predict(X_test)[0]


class Translator:
    def __init__(self):
        self.dictionary = {
            "French": {
                "bonjour": "hello",
                "merci": "thank you",
                "couleur": "color",
                "favourite": "favourite"
            },
            "Spanish": {
                "hola": "hello",
                "gracias": "thank you",
                "color": "color",
                "favorito": "favorite"
            },
            "German": {
                "hallo": "hello",
                "danke": "thank you",
                "farbe": "color",
                "lieblings": "favorite"
            }
        }

    def translate(self, text, source_language):
        words = text.lower().split()
        mapping = self.dictionary.get(source_language, {})
        translated = [mapping.get(w, w) for w in words]
        return " ".join(translated)


class UKEnglishNormaliser:
    def __init__(self):
        # US -> UK spelling replacements
        self.replacements = {
            "color": "colour",
            "favorite": "favourite",
            "organize": "organise",
            "favor": "favour"
        }

    def normalise(self, text):
        for us, uk in self.replacements.items():
            text = text.replace(us, uk)
        return text


class SpeechTranslationPipeline:
    def __init__(self):
        self.detector = LanguageDetectorML()
        self.translator = Translator()
        self.normaliser = UKEnglishNormaliser()

    def run(self, transcript):
        language = self.detector.predict(transcript)
        translated = self.translator.translate(transcript, language)
        final_text = self.normaliser.normalise(translated)
        return language, final_text


# Streamlit app
st.set_page_config(page_title="Multi-language Speech Translator", layout="centered")
st.title("Multi-language Speech Translator")

pipeline = SpeechTranslationPipeline()
input_text = st.text_area("Enter speech transcript:", height=150)

if st.button("Translate"):
    if input_text.strip():
        detected_language, output = pipeline.run(input_text)
        st.write(f"**Detected language:** {detected_language}")
        st.text_area("UK English output:", value=output, height=120)
    else:
        st.warning("Please enter a speech transcript.")
