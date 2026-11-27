import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import altair as alt


# Dataset Generation
class Dataset:
    def __init__(self, n_users=5, n_items=10, n_features=5):
        self.n_users = n_users
        self.n_items = n_items
        self.n_features = n_features
        self.users = [f"User {i+1}" for i in range(n_users)]
        self.items = [f"Item {i+1}" for i in range(n_items)]
        self.item_features = self.generate_item_features()
    
    def generate_item_features(self):
        np.random.seed(42)
        return pd.DataFrame(
            np.random.rand(self.n_items, self.n_features),
            index=self.items,
            columns=[f"Feature {i+1}" for i in range(self.n_features)]
        )


# Embedding Model
class EmbeddingRecommender:
    def __init__(self, item_features):
        self.item_features = item_features
        self.scaler = StandardScaler()
        self.item_embeddings = self.compute_embeddings()
    
    def compute_embeddings(self):
        scaled = self.scaler.fit_transform(self.item_features)
        return pd.DataFrame(scaled, index=self.item_features.index, columns=self.item_features.columns)
    
    def recommend(self, user_vector, top_k=5):
        similarities = cosine_similarity([user_vector], self.item_embeddings.values)[0]
        recommendations = pd.DataFrame({
            "Item": self.item_embeddings.index,
            "Score": similarities
        }).sort_values(by="Score", ascending=False)
        return recommendations.head(top_k)


# Trainer
class Trainer:
    def __init__(self, dataset, recommender):
        self.dataset = dataset
        self.recommender = recommender
        self.user_vectors = self.generate_user_vectors()
    
    def generate_user_vectors(self):
        vectors = {}
        for user in self.dataset.users:
            seed = sum(ord(c) for c in user)
            rng = np.random.default_rng(seed)
            vectors[user] = rng.random(self.dataset.n_features)
        return vectors
    
    def get_user_vector(self, user_name):
        return self.user_vectors[user_name]


# Streamlit UI
st.title("Personalised Recommender with Embeddings")

dataset = Dataset()
recommender = EmbeddingRecommender(dataset.item_features)
trainer = Trainer(dataset, recommender)

st.subheader("Items Feature Table")
st.dataframe(dataset.item_features)

user_selection = st.selectbox("Select User:", dataset.users)
if st.button("Get Recommendations"):
    user_vector = trainer.get_user_vector(user_selection)
    recs = recommender.recommend(user_vector, top_k=5)
    
    st.subheader(f"Top Recommendations for {user_selection}")
    st.table(recs)

    # Visualisation: Similarity Bar Chart
    chart = alt.Chart(recs).mark_bar().encode(
        x=alt.X('Score:Q', title='Similarity Score'),
        y=alt.Y('Item:N', sort='-x', title='Item'),
        tooltip=['Item', 'Score']
    ).properties(width=600, height=300, title="Recommendation Similarity Scores")
    
    st.altair_chart(chart)
