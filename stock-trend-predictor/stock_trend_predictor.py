import numpy as np


class StockDataGenerator:
    def __init__(self, days: int = 120, seed: int = 42):
        self.days = days
        self.seed = seed

    def generate(self):
        np.random.seed(self.seed)
        x = np.arange(self.days, dtype=float)
        trend = 0.2 * x
        noise = np.random.normal(0, 2, self.days)
        y = 50 + trend + noise
        return x, y


class LinearRegressionModel:
    def __init__(self):
        self.weight = 0.0
        self.bias = 0.0

    def predict(self, x):
        return self.weight * x + self.bias


class Trainer:
    def __init__(self, model, learning_rate=0.05, epochs=1000):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, x, y):
        n = len(x)

        x_norm = (x - x.mean()) / x.std()

        for _ in range(self.epochs):
            y_pred = self.model.predict(x_norm)

            error = y_pred - y
            dw = (2/n) * np.dot(error, x_norm)
            db = (2/n) * np.sum(error)

            self.model.weight -= self.learning_rate * dw
            self.model.bias -= self.learning_rate * db

        self.x_mean = x.mean()
        self.x_std = x.std()

    def predict(self, x):
        x_norm = (x - self.x_mean) / self.x_std
        return self.model.predict(x_norm)


class Evaluator:
    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


def main():
    generator = StockDataGenerator()
    x, y = generator.generate()

    model = LinearRegressionModel()
    trainer = Trainer(model)
    trainer.train(x, y)

    predictions = trainer.predict(x)
    mse = Evaluator.mean_squared_error(y, predictions)

    trend_direction = "upward" if model.weight > 0 else "downward"

    print("\n Stock Trend Prediction (Linear Regression)")
    print("-----------------------------------------")
    print(f"Learned weight (slope): {model.weight:.4f}")
    print(f"Learned bias (intercept): {model.bias:.4f}")
    print(f"Mean squared error: {mse:.2f}")
    print(f"Predicted trend direction: {trend_direction}\n")


if __name__ == "__main__":
    main()