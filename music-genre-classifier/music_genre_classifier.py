import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


class DataGenerator:
    def __init__(self, n_samples=1000, n_features=20, n_classes=5, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.random_state = random_state

    def generate(self):
        np.random.seed(self.random_state)
        X = np.zeros((self.n_samples, self.n_features))
        y = np.zeros(self.n_samples, dtype=int)

        samples_per_class = self.n_samples // self.n_classes

        for cls in range(self.n_classes):
            start = cls * samples_per_class
            end = start + samples_per_class
            # Each class has a different mean in feature space
            mean = np.random.rand(self.n_features) * (cls + 1)
            X[start:end] = np.random.randn(samples_per_class, self.n_features) + mean
            y[start:end] = cls

        # Shuffle dataset
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)
        return X[indices], y[indices]


class MusicGenreModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)


class Evaluator:
    @staticmethod
    def evaluate(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        return acc, report


def main():
    print("\nGenerating synthetic patterned music features...")
    data_gen = DataGenerator()
    X, y = data_gen.generate()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining classifier...")
    model = MusicGenreModel()
    model.train(X_train, y_train)

    print("\nEvaluating model...")
    y_pred = model.predict(X_test)
    acc, report = Evaluator.evaluate(y_test, y_pred)

    print(f"\nAccuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()