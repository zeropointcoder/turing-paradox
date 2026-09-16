import numpy as np
from python_speech_features import mfcc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import random


class DataGenerator:
    def __init__(self, commands=None, samples_per_command=50, fs=8000, duration=1):
        self.commands = commands or ["up", "down", "left", "right", "stop", "go"]
        self.samples_per_command = samples_per_command
        self.fs = fs
        self.duration = duration
        self.dataset = {}

    def generate_sine_wave(self, freq):
        t = np.linspace(0, self.duration, int(self.fs * self.duration), endpoint=False)
        wave = 0.5 * np.sin(2 * np.pi * freq * t)
        return (wave * 32767).astype(np.int16)

    def create_dataset(self):
        freqs = np.linspace(200, 800, len(self.commands))
        for cmd, freq in zip(self.commands, freqs):
            self.dataset[cmd] = []
            for _ in range(self.samples_per_command):
                audio = self.generate_sine_wave(freq + np.random.randint(-20, 20))
                self.dataset[cmd].append(audio)

    def get_features_labels(self):
        X, y = [], []
        for cmd, audios in self.dataset.items():
            for audio in audios:
                features = mfcc(audio, samplerate=self.fs)
                X.append(features.mean(axis=0))
                y.append(cmd)
        return np.array(X), np.array(y)


class VoiceModel:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, features):
        return self.model.predict([features.mean(axis=0)])[0]

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


class VoiceCLI:
    def __init__(self):
        self.data_gen = DataGenerator()
        self.model = VoiceModel()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def setup(self):
        self.data_gen.create_dataset()
        X, y = self.data_gen.get_features_labels()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.train(self.X_train, self.y_train)
        self.model.evaluate(self.X_test, self.y_test)

    def test_command(self):
        cmd = random.choice(list(self.data_gen.dataset.keys()))
        audio = random.choice(list(self.data_gen.dataset[cmd]))
        features = mfcc(audio, samplerate=self.data_gen.fs)
        predicted = self.model.predict(features)
        print(f"Synthetic sample for '{cmd}' recognised as: {predicted}")


if __name__ == "__main__":
    cli = VoiceCLI()
    cli.setup()
    while True:
        cmd = input("\nPress Enter to test a synthetic command (or type 'exit' to quit): ")
        if cmd.lower() == "exit":
            break
        cli.test_command()
    print("\n")