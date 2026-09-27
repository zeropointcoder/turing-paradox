import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


class DatasetBuilder:
    def build(self):
        uk_english = [
            "I prefer the colour blue.",
            "The theatre is located in the city centre.",
            "He organised the programme carefully.",
            "She travelled by aeroplane.",
            "The cheque was sent yesterday.",
            "My favourite restaurant is near the harbour.",
            "The neighbour parked outside the house.",
            "We need to buy some petrol.",
            "The lift is on the second floor.",
            "She is studying at university.",
            "The shop sells biscuits and sweets.",
            "He apologised for being late.",
            "The football team played brilliantly.",
            "Please queue outside the building.",
            "The holiday begins next Monday.",
            "I need to renew my driving licence.",
            "The grey jacket is on the chair.",
            "She learnt French at school.",
            "The centre of town is very busy.",
            "He is wearing a jumper today.",
            "The pavement was wet after the rain.",
            "We travelled by underground.",
            "The programme starts at eight o'clock.",
            "I bought a packet of crisps.",
            "The car is parked near the kerb.",
            "She wrote her name on the cheque.",
            "The university library closes at six.",
            "He has a large collection of books.",
            "The weather was brilliant yesterday.",
            "My favourite colour is turquoise."
        ]

        non_english = [
            "La biblioteca está cerrada hoy.",
            "Das Wetter ist heute sehr schön.",
            "La macchina è parcheggiata fuori.",
            "El niño come una manzana.",
            "Le chat dort sur le canapé.",
            "La maison est près de la rivière.",
            "Mi hermano vive en Madrid.",
            "Das Kind spielt im Garten.",
            "Je vais au marché demain.",
            "Il cane corre nel parco.",
            "La comida está sobre la mesa.",
            "Wir fahren morgen nach Berlin.",
            "Elle aime beaucoup la musique.",
            "Il libro è molto interessante.",
            "Los estudiantes están en la escuela.",
            "Le train arrive à midi.",
            "Meine Schwester arbeitet im Krankenhaus.",
            "La voiture est rouge.",
            "El profesor explica la lección.",
            "Il sole splende oggi.",
            "Nous habitons dans une grande ville.",
            "Das Haus hat drei Fenster.",
            "Mi padre cocina la cena.",
            "La ragazza legge un libro.",
            "Les enfants jouent dans le jardin.",
            "El perro duerme en la habitación.",
            "Ich trinke gerne Kaffee.",
            "La fenêtre est ouverte.",
            "Gli amici vanno al ristorante.",
            "Nous allons voyager en été."
        ]

        texts = uk_english + non_english
        labels = ["UK English"] * len(uk_english) + ["Non-English"] * len(non_english)
        return texts, labels
    

class LanguageModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("clf", MultinomialNB())
        ])

    def train(self, x, y):
        self.pipeline.fit(x, y)

    def predict(self, text):
        return self.pipeline.predict([text])[0]


class AppController:
    def __init__(self):
        self.dataset = DatasetBuilder()
        self.model = LanguageModel()

    def run(self):
        x, y = self.dataset.build()
        self.model.train(x, y)

        st.title("Language Detection - UK English")
        user_input = st.text_area("Enter text for detection:")

        if st.button("Detect Language") and user_input.strip():
            prediction = self.model.predict(user_input)
            if prediction == "UK English":
                st.success(f"Prediction: {prediction}")
            else:
                st.warning(f"Prediction: {prediction}")


if __name__ == "__main__":
    AppController().run()