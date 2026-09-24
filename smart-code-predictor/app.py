import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# Dataset Generation
class CodeDataset:
    def __init__(self):
        self.examples = [
            "def add(a, b):\n    return a + b\n",
            "for i in range(10):\n    print(i)\n",
            "if x > 0:\n    print('Positive')\nelse:\n    print('Negative')\n",
            "class Person:\n    def __init__(self, name):\n        self.name = name\n",
            "while n > 0:\n    n -= 1\n    print(n)\n"
        ]
        self.vocab = sorted(list(set(''.join(self.examples))))
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char = {i: ch for i, ch in enumerate(self.vocab)}

    def get_training_data(self, seq_len=10):
        inputs, targets = [], []
        for example in self.examples:
            for i in range(len(example) - seq_len):
                seq = example[i:i+seq_len]
                next_char = example[i+seq_len]
                inputs.append([self.char2idx[c] for c in seq])
                targets.append(self.char2idx[next_char])
        return torch.tensor(inputs), torch.tensor(targets)

# Model
class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        x = self.embedding(x)
        out, h = self.rnn(x, h)
        out = self.fc(out[:, -1, :])
        return out, h


# Trainer
class Trainer:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.criterion = nn.CrossEntropyLoss()
        self.optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    def train(self, epochs=50):
        X, y = self.dataset.get_training_data()
        for epoch in range(epochs):
            self.model.train()
            self.optimiser.zero_grad()
            out, _ = self.model(X)
            loss = self.criterion(out, y)
            loss.backward()
            self.optimiser.step()


# Evaluator / Predictor
class CodePredictor:
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset

    def predict(self, start_seq, length=50):
        self.model.eval()
        seq = [self.dataset.char2idx.get(c, 0) for c in start_seq]
        result = start_seq
        h = None
        for _ in range(length):
            x = torch.tensor([seq[-10:]])
            out, h = self.model(x, h)
            next_idx = torch.argmax(F.softmax(out, dim=1), dim=1).item()
            result += self.dataset.idx2char[next_idx]
            seq.append(next_idx)
        return result

# Streamlit UI
st.title("Smart Code Predictor")

dataset = CodeDataset()
model = CharRNN(vocab_size=len(dataset.vocab))
trainer = Trainer(model, dataset)
predictor = CodePredictor(model, dataset)

if st.button("Train Model"):
    trainer.train(epochs=100)
    st.success("Model trained!")

start_seq = st.text_area("Start your code sequence (Python):", "def ")
length = st.slider("Prediction Length:", 20, 200, 50)

if st.button("Predict Code"):
    if not hasattr(model, 'trained'):
        trainer.train(epochs=100)
        model.trained = True
    prediction = predictor.predict(start_seq=start_seq, length=length)
    st.code(prediction, language="python")
