import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class SyntheticDataGenerator:
    def __init__(self, users=50, products=30, seed=42):
        self.users = users
        self.products = products
        self.seed = seed

    def generate(self):
        rng = np.random.default_rng(self.seed)
        interactions = rng.integers(0, 6, size=(self.users, self.products))
        users = [f"User_{i+1}" for i in range(self.users)]
        products = [f"Product_{j+1}" for j in range(self.products)]
        return pd.DataFrame(interactions, index=users, columns=products)


class RecommenderModel:
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_matrix = None
        self.similarity = None

    def train(self, data):
        self.user_matrix = self.svd.fit_transform(data)
        self.similarity = cosine_similarity(self.user_matrix)

    def recommend(self, user_index, data, top_n=5):
        scores = self.similarity[user_index]
        similar_users = np.argsort(scores)[::-1][1:]
        weighted = np.zeros(data.shape[1])

        for idx in similar_users[:10]:
            weighted += scores[idx] * data.iloc[idx].values

        recommended_indices = np.argsort(weighted)[::-1][:top_n]
        return data.columns[recommended_indices]


class RecommendationApp:
    def __init__(self):
        self.data_generator = SyntheticDataGenerator()
        self.data = self.data_generator.generate()
        self.model = RecommenderModel()
        self.model.train(self.data)

    def run(self):
        st.title("Product Recommendation System")

        user = st.selectbox("Select a user", self.data.index)
        top_n = st.slider("Number of recommendations", 1, 10, 5)

        user_index = self.data.index.get_loc(user)
        recommendations = self.model.recommend(user_index, self.data, top_n)

        st.subheader("Recommended products")
        for product in recommendations:
            st.write(product)

        with st.expander("View interaction data"):
            st.dataframe(self.data)


if __name__ == "__main__":
    RecommendationApp().run()