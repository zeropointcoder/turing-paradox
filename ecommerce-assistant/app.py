import streamlit as st
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Product Dataset
PRODUCTS = [
    {"name": "Eco-Friendly Winter Coat", "category": "clothing", "price": "£120"},
    {"name": "Noise-Cancelling Headphones", "category": "electronics", "price": "£80"},
    {"name": "Gaming Laptop", "category": "electronics", "price": "£999"},
    {"name": "Organic Tea Set", "category": "food", "price": "£25"},
    {"name": "UK Brand Sneakers", "category": "footwear", "price": "£60"},
    {"name": "Smartphone Case", "category": "electronics", "price": "£15"},
    {"name": "Winter Gloves", "category": "clothing", "price": "£20"},
    {"name": "Bluetooth Speaker", "category": "electronics", "price": "£45"},
    {"name": "Herbal Skincare Kit", "category": "beauty", "price": "£35"},
    {"name": "Leather Wallet", "category": "accessories", "price": "£50"},
    {"name": "Cotton Summer Dress", "category": "clothing", "price": "£40"},
    {"name": "Running Shoes", "category": "footwear", "price": "£70"},
    {"name": "Stainless Steel Water Bottle", "category": "home", "price": "£18"},
    {"name": "Organic Honey", "category": "food", "price": "£12"},
    {"name": "Digital Alarm Clock", "category": "electronics", "price": "£22"},
    {"name": "Wool Scarf", "category": "clothing", "price": "£25"},
    {"name": "Yoga Mat", "category": "fitness", "price": "£30"},
    {"name": "Electric Kettle", "category": "home", "price": "£28"},
    {"name": "UK Flag Backpack", "category": "accessories", "price": "£55"},
    {"name": "Chocolate Gift Box", "category": "food", "price": "£15"},
    {"name": "Smartwatch", "category": "electronics", "price": "£199"},
    {"name": "Hair Care Set", "category": "beauty", "price": "£40"},
    {"name": "Winter Boots", "category": "footwear", "price": "£85"},
    {"name": "Desk Lamp", "category": "home", "price": "£20"},
    {"name": "Fitness Tracker", "category": "fitness", "price": "£60"},
]


# ML Chatbot Class
class MLBasedECommerceChatBot:
    def __init__(self, products: List[Dict[str, str]]):
        self.products = products
        self.memory: List[Dict[str, str]] = []
        self.vectorizer = TfidfVectorizer()
        self.product_texts = [p["name"] + " " + p["category"] for p in products]
        self.product_vectors = self.vectorizer.fit_transform(self.product_texts)

    def add_to_memory(self, user_input: str, bot_response: str):
        self.memory.append({"user": user_input, "bot": bot_response})

    def generate_response(self, user_input: str) -> str:
        # Transform user query
        query_vector = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(query_vector, self.product_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:3]  # top 3 products

        matches = [f"{self.products[i]['name']} ({self.products[i]['price']})" for i in top_indices if similarities[i] > 0]

        # Memory-aware hint
        previous_categories = [
            p["category"] for msg in self.memory for p in self.products
            if p["name"].lower() in msg["user"].lower() or p["category"].lower() in msg["user"].lower()
        ]
        memory_hint = f" Previously, you asked about: {', '.join(set(previous_categories))}." if previous_categories else ""

        if matches:
            response = f"I suggest these products: {', '.join(matches)}.{memory_hint}"
        else:
            response = "No close matches found. You might like: " + ", ".join(f"{p['name']} (£{p['price']})" for p in self.products[:3]) + memory_hint

        self.add_to_memory(user_input, response)
        return response

    def get_history(self) -> List[Dict[str, str]]:
        return self.memory


# Streamlit Interface
st.set_page_config(page_title="ML E-Commerce Assistant", layout="wide")
st.title("ML-Powered E-Commerce Assistant")

if "chatbot" not in st.session_state:
    st.session_state.chatbot = MLBasedECommerceChatBot(PRODUCTS)

user_input = st.text_input("Ask about products, categories, or deals:")

if st.button("Send") and user_input:
    response = st.session_state.chatbot.generate_response(user_input)
    st.text_area("Assistant:", value=response, height=150)

# Display conversation history
st.markdown("### Conversation History")
history = st.session_state.chatbot.get_history()
for msg in history:
    st.markdown(f"**User:** {msg['user']}")
    st.markdown(f"**Assistant:** {msg['bot']}")
