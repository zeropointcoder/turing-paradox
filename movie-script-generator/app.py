import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
import random


# Corpus class
class ScriptCorpus:
    def __init__(self):
        self.text = self._load_text()
        self.chars = sorted(list(set(self.text)))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}

    def _load_text(self):
        return """
        INT. LONDON FLAT - NIGHT
        Rain taps against the window.

        SARAH
        I never thought it would end like this.

        JAMES
        Nothing ever ends. It only changes.

        EXT. CITY STREET - DAY
        Crowds rush past. A siren wails.

        DETECTIVE
        Everyone has something to hide.

        SARAH
        I just wanted the truth.

        FADE OUT.
        """

    def encode(self, text):
        return [self.char2idx[c] for c in text]

    def decode(self, indices):
        return ''.join([self.idx2char[i] for i in indices])


# Character-level LSTM
class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(vocab_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


# Trainer class
class ScriptTrainer:
    def __init__(self, corpus, seq_length=50, lr=0.005, epochs=10):  # reduced to 10 epochs
        self.corpus = corpus
        self.seq_length = seq_length
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device("cpu")

        self.vocab_size = len(corpus.chars)
        self.model = CharLSTM(self.vocab_size).to(self.device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

    def _create_sequences(self):
        text_indices = self.corpus.encode(self.corpus.text)
        sequences = []
        targets = []
        for i in range(0, len(text_indices) - self.seq_length):
            seq = text_indices[i:i+self.seq_length]
            target = text_indices[i+1:i+self.seq_length+1]
            sequences.append(seq)
            targets.append(target)
        return sequences, targets

    def train(self, st_placeholder=None):
        sequences, targets = self._create_sequences()
        for epoch in range(self.epochs):
            total_loss = 0

            for i in range(len(sequences)):
                hidden = self.model.init_hidden(1)
                hidden = tuple([h.detach() for h in hidden])

                seq_tensor = torch.zeros(1, self.seq_length, self.vocab_size)
                for t, idx in enumerate(sequences[i]):
                    seq_tensor[0, t, idx] = 1.0

                target_tensor = torch.tensor(targets[i]).unsqueeze(0)

                self.optimiser.zero_grad()
                output, hidden = self.model(seq_tensor, hidden)
                loss = self.criterion(output.view(-1, self.vocab_size), target_tensor.view(-1))
                loss.backward()
                self.optimiser.step()
                total_loss += loss.item()

            if st_placeholder:
                st_placeholder.text(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(sequences):.4f}")


# Script Generator class
class ScriptGenerator:
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.corpus = trainer.corpus

    def generate_script(self, length=300):
        self.model.eval()
        start_idx = random.randint(0, len(self.corpus.text) - self.trainer.seq_length - 1)
        seq = self.corpus.encode(self.corpus.text[start_idx:start_idx+self.trainer.seq_length])
        generated = seq.copy()
        hidden = self.model.init_hidden(1)

        for _ in range(length):
            x = torch.zeros(1, self.trainer.seq_length, self.trainer.vocab_size)
            for t, idx in enumerate(seq):
                x[0, t, idx] = 1.0
            output, hidden = self.model(x, hidden)
            probs = F.softmax(output[0, -1], dim=0).detach().numpy()
            next_idx = np.random.choice(len(probs), p=probs)
            generated.append(next_idx)
            seq = seq[1:] + [next_idx]

        return self.corpus.decode(generated)


# Streamlit app
class App:
    def __init__(self):
        st.set_page_config(page_title="Movie Script Generator", layout="centered")
        st.title("Movie Script Generator (Character-level LSTM)")

    def run(self):
        trainer = ScriptTrainer(ScriptCorpus())

        if st.button("Train Model (takes a few seconds)"):
            with st.spinner("Training in progress..."):  # spinner loader
                loader = st.empty()  # placeholder for epoch test
                trainer.train(st_placeholder=loader)
                loader.empty()
            st.success("Training completed!")

        length = st.slider("Script length (characters)", 100, 500, 300)
        if st.button("Generate Script"):
            generator = ScriptGenerator(trainer)
            script = generator.generate_script(length)
            st.text_area("Generated Script", script, height=320)


# Run app
if __name__ == "__main__":
    App().run()