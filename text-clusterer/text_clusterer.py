import sys
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class DocumentDataset:
    def __init__(self, documents: List[str]):
        self.documents = documents

    @staticmethod
    def generate_sample_docs() -> List[str]:
        # UK English, themed and slightly longer for better clustering
        docs = [
            # Nature & outdoors
            "The colour of the sky changes from azure to golden at sunset.",
            "Gardening in spring brings vibrant colours to the garden beds.",
            "Birdwatchers gather in the countryside to observe rare species.",
            "A walk along the riverbank provides a refreshing view of nature.",
            
            # Arts & literature
            "The theatre performance received a standing ovation from the audience.",
            "Reading classic novels expands the mind and enhances creativity.",
            "She painted a landscape inspired by the English countryside.",
            "The art exhibition showcased contemporary UK artists.",
            
            # Everyday life & UK culture
            "Organising events requires careful planning and attention to detail.",
            "Cooking a proper meal involves selecting fresh ingredients.",
            "He wore a lovely jumper to keep warm during winter evenings.",
            "The council approved new regulations for public transport routes.",
            "Football matches are watched passionately by fans across the UK.",
        ]
        return docs


class DocumentClusterer:
    def __init__(self, n_clusters: int = 3):
        self.n_clusters = n_clusters
        self.vectoriser = TfidfVectorizer()
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42)

    def fit(self, documents: List[str]):
        X = self.vectoriser.fit_transform(documents)
        self.model.fit(X)
        labels = self.model.labels_
        score = silhouette_score(X, labels)
        return labels, score


class CLI:
    @staticmethod
    def main():
        print("\n--- Document Clustering CLI ---\n")
        dataset = DocumentDataset(DocumentDataset.generate_sample_docs())
        clusterer = DocumentClusterer(n_clusters=3)
        labels, score = clusterer.fit(dataset.documents)
        print("Cluster assignments:")
        for doc, label in zip(dataset.documents, labels):
            print(f"\n[Cluster {label}] {doc}")
        print(f"\nSilhouette Score: {score:.3f}\n")
        print("Clustering completed successfully.\n")


if __name__ == "__main__":
    CLI.main()