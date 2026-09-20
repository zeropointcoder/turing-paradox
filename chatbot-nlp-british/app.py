import re
import string
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


class TextNormaliser:
    def __init__(self):
        self.patterns = [
            (r"\bcolour\b", "colour"),
            (r"\bfavourite\b", "favourite"),
            (r"\borganise\b", "organise")
        ]

    def normalise(self, text: str) -> str:
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation)) # remove punctuation
        for pattern, replacement in self.patterns:
            text = re.sub(pattern, replacement, text)
        return text


class MLModel:
    def __init__(self):
        # Offline dataset of intents with more variations
        self.intents = {
            "greeting": ["hello", "hi", "hey", "good morning", "good afternoon", "hiya", "hello there"],
            "farewell": ["bye", "goodbye", "see you", "cheerio", "bye bye", "take care"],
            "how_are_you": ["how are you", "how's it going", "how do you do", "how are you doing", "how are you today"],
            "name_query": ["your name", "who are you", "what is your name", "may I know your name", "can you tell me your name"],
            "weather_query": ["weather", "is it raining", "is it sunny", "forecast", "what's the weather like", "will it rain today"],
        }

        self.responses = {
            "greeting": "Hello! How can I help you today?",
            "farewell": "Goodbye! Have a lovely day.",
            "how_are_you": "I'm doing well, thank you. How about you?",
            "name_query": "I'm a UK-English chatbot built for demonstration purposes.",
            "weather_query": "I can't check live weather, but always carry an umbrella in the UK.",
            "unknown": "I'm not sure I understand. Could you rephrase that?",
        }

        # Prepare training data
        self.train_texts = []
        self.train_labels = []
        for intent, phrases in self.intents.items():
            self.train_texts.extend(phrases)
            self.train_labels.extend([intent] * len(phrases))

        # TF-IDF vectorizer and Naives Bayes classifier
        self.vectorizer = TfidfVectorizer()
        X_train = self.vectorizer.fit_transform(self.train_texts)
        self.classifier = MultinomialNB()
        self.classifier.fit(X_train, self.train_labels)

    def predict(self, text: str) -> str:
        X_test = self.vectorizer.transform([text])
        probs = self.classifier.predict_proba(X_test)[0]
        intent_pred = self.classifier.classes_[probs.argmax()]
        # Confidence threshold: if low, return unknown
        if probs.max() < 0.3:
            return self.responses["unknown"]
        return self.responses.get(intent_pred, self.responses["unknown"])


class ChatbotEngine:
    def __init__(self):
        self.normaliser = TextNormaliser()
        self.model = MLModel()

    def get_response(self, user_input: str) -> str:
        cleaned_text = self.normaliser.normalise(user_input)
        return self.model.predict(cleaned_text)


class ChatbotUI:
    def __init__(self):
        self.engine = ChatbotEngine()

    def render(self):
        st.set_page_config(page_title="British Chatbot", layout="centered")
        st.title("ML-Based UK Chatbot")

        if "history" not in st.session_state:
            st.session_state.history = []

        user_input = st.text_input("You:", placeholder="Type your message here")

        if user_input:
            response = self.engine.get_response(user_input)
            st.session_state.history.append(("You", user_input))
            st.session_state.history.append(("Bot", response))

        for speaker, message in st.session_state.history:
            st.markdown(f"**{speaker}:** {message}")


if __name__ == "__main__":
    ChatbotUI().render()