import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


class CustomerDataGenerator:
    def __init__(self, n_samples: int, random_state: int = 42):
        self.n_samples = n_samples
        self.random_state = random_state

    def generate(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)

        age = rng.integers(18, 70, self.n_samples)
        annual_income = rng.normal(35000, 15000, self.n_samples).clip(12000, 120000)
        spending_score = rng.integers(1, 100, self.n_samples)

        return pd.DataFrame({
            "Age": age,
            "Annual Income (£)": annual_income,
            "Spending Score": spending_score
        })


class KMeansModel:
    def __init__(self, n_clusters: int, random_state: int = 42):
        self.model = KMeans(
            n_clusters=n_clusters,
            n_init="auto",
            random_state=random_state
        )

    def fit_predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.model.fit_predict(data)

    def inertia(self) -> float:
        return self.model.inertia_


class Trainer:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def train(self, n_clusters: int):
        model = KMeansModel(n_clusters)
        labels = model.fit_predict(self.data)
        return model, labels


class Evaluator:
    @staticmethod
    def silhouette(data: pd.DataFrame, labels: np.ndarray) -> float:
        return silhouette_score(data, labels)


class Visualiser:
    @staticmethod
    def plot_clusters(data: pd.DataFrame, labels: np.ndarray):
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            data["Annual Income (£)"],
            data["Spending Score"],
            c=labels
        )
        ax.set_xlabel("Annual Income (£)")
        ax.set_ylabel("Spending Score")
        ax.set_title("Customer Segments")
        return fig


def main():
    st.title("Customer Segmentation with K-means")

    st.sidebar.header("Configuration")
    n_samples = st.sidebar.slider("Number of customers", 100, 2000, 500)
    n_clusters = st.sidebar.slider("Number of clusters", 2, 10, 4)

    generator = CustomerDataGenerator(n_samples)
    data = generator.generate()

    trainer = Trainer(data)
    model, labels = trainer.train(n_clusters)

    silhouette = Evaluator.silhouette(data, labels)

    data["Cluster"] = labels

    st.subheader("Sample Data")
    st.dataframe(data.head())

    st.subheader("Clustering Result")
    fig = Visualiser.plot_clusters(data, labels)
    st.pyplot(fig, width=700)

    st.subheader("Evaluation")
    st.metric("Silhouette Score", f"{silhouette:.3f}")
    st.metric("Within-cluster Sum of Squares", f"{model.inertia():,.0f}")


if __name__ == "__main__":
    main()