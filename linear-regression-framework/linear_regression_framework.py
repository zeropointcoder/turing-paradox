import argparse
import numpy as np
import matplotlib.pyplot as plt


# Generate synthetic data
class DataGenerator:
    def __init__(self, samples=500, noise=0.5, seed=42):
        self.samples = samples
        self.noise = noise
        self.seed = seed

    def generate(self):
        np.random.seed(self.seed)
        x = np.random.uniform(-5, 5, self.samples)
        y = 3.5 * x + 2 + np.random.normal(0, self.noise, self.samples)
        return x.reshape(-1, 1), y

    def train_test_split(self, x, y, test_ratio=0.2):
        n = len(y)
        indices = np.arange(n)
        np.random.shuffle(indices)
        split = int(n * (1 - test_ratio))
        train_idx, test_idx = indices[:split], indices[split:]
        return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


# Linear Regression Model
class LinearRegressionModel:
    def __init__(self):
        self.weight = 0.0
        self.bias = 0.0

    def predict(self, x):
        return self.weight * x.flatten() + self.bias


# Train using Gradient Descent
class Trainer:
    def __init__(self, model, lr=0.01, epochs=1000):
        self.model = model
        self.lr = lr
        self.epochs = epochs

    def train(self, x, y):
        n = len(y)
        for _ in range(self.epochs):
            preds = self.model.predict(x)
            dw = (-2 / n) * np.sum((y - preds) * x.flatten())
            db = (-2 / n) * np.sum(y - preds)
            self.model.weight -= self.lr * dw
            self.model.bias -= self.lr * db


# Evaluator for metrics
class Evaluator:
    def __init__(self, model):
        self.model = model

    def mse(self, x, y):
        preds = self.model.predict(x)
        return np.mean((y - preds) ** 2)


# Visualisation
class Visualiser:
    @staticmethod
    def plot_predictions(x, y_true, y_pred):
        plt.scatter(x, y_true, label='True', alpha=0.6)
        plt.scatter(x, y_pred, label='Predicted', alpha=0.6)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Linear Regression Predictions')
        plt.legend()
        plt.show()


# Main CLI
def main():
    parser = argparse.ArgumentParser(description="OpenLearn CLI Trainer")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    # Data
    data_gen = DataGenerator(samples=args.samples)
    x, y = data_gen.generate()
    x_train, x_test, y_train, y_test = data_gen.train_test_split(x, y)

    # Model
    model = LinearRegressionModel()
    trainer = Trainer(model, lr=args.lr, epochs=args.epochs)
    trainer.train(x_train, y_train)

    # Evaluation
    evaluator = Evaluator(model)
    train_error = evaluator.mse(x_train, y_train)
    test_error = evaluator.mse(x_test, y_test)

    print("\nTraining complete!\n")
    print(f"Weight: {model.weight:.3f}")
    print(f"Bias: {model.bias:.3f}")
    print(f"Train MSE: {train_error:.4f}")
    print(f"Test MSE: {test_error:.4f}\n")

    # Visualisation
    y_pred = model.predict(x_test)
    Visualiser.plot_predictions(x_test, y_test, y_pred)


if __name__ == "__main__":
    main()