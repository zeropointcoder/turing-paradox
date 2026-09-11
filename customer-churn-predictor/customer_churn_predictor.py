import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class ChurnDataset:
    def __init__(self, samples=1000, features=8, random_state=42):
        self.samples = samples
        self.features = features
        self.random_state = random_state

    def load(self):
        X, y = make_classification(
            n_samples=self.samples,
            n_features=self.features,
            n_informative=5,
            n_redundant=1,
            n_classes=2,
            weights=[0.65, 0.35],
            random_state=self.random_state
        )
        columns = [f"feature_{i}" for i in range(self.features)]
        data = pd.DataFrame(X, columns=columns)
        data["churn"] = y
        return data


class ChurnModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)


class Trainer:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model

    def run(self):
        data = self.dataset.load()
        X = data.drop("churn", axis=1)
        y = data["churn"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.model.train(X_train, y_train)
        predictions = self.model.predict(X_test)

        return Evaluator.evaluate(y_test, predictions)

class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred)
        }


def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction CLI")
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()

    dataset = ChurnDataset(samples=args.samples)
    model = ChurnModel()
    trainer = Trainer(dataset, model)

    results = trainer.run()

    print("\n")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print("\n")


if __name__ == "__main__":
    main()