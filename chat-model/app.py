import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# Dataset
class UKChatDataset:
    def __init__(self):
        self.sentences = self._build_corpus()
        vocab = sorted(set(" ".join(self.sentences).split()))
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = {i: w for w, i in self.stoi.items()}
        self.vocab_size = len(vocab)

    def encode(self, text):
        return [self.stoi[w] for w in text.split() if w in self.stoi]

    def decode(self, tokens):
        return " ".join(self.itos[t] for t in tokens)

    def _build_corpus(self):
        base = [
            "Hello how are you today",
            "I am quite well thank you",
            "What would you like to discuss",
            "The weather is rather pleasant today",
            "I do prefer tea over coffee",
            "That sounds like a brilliant idea",
            "I am not particularly keen on that",
            "Shall we talk about something interesting",
        ]
        return base * 300


# Model
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, dim=96, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                activation="gelu",
                batch_first=True
            ) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.fc(x)


# Training
class Trainer:
    def __init__(self, model, dataset):
        self.model = model
        self.ds = dataset
        self.optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
        self.data = self._prepare_data()

    def _prepare_data(self):
        sequences = []
        for s in self.ds.sentences:
            enc = self.ds.encode(s)
            if len(enc) > 3:
                sequences.append(torch.tensor(enc))
        return sequences

    def train(self, steps=500, block_size=8):
        self.model.train()
        for _ in range(steps):
            seq = random.choice(self.data)
            if len(seq) <= block_size:
                continue
            i = random.randint(0, len(seq) - block_size - 1)
            x = seq[i:i+block_size].unsqueeze(0)
            y = seq[i+1:i+block_size+1].unsqueeze(0)

            logits = self.model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()


# Generation
class Generator:
    def __init__(self, model, dataset):
        self.model = model
        self.ds = dataset

    def generate(self, prompt, max_len=20, temperature=0.7, top_k=10):
        self.model.eval()
        tokens = self.ds.encode(prompt)
        if not tokens:
            tokens = random.sample(list(self.ds.stoi.values()), 2)

        tokens = torch.tensor(tokens).unsqueeze(0)

        for _ in range(max_len):
            logits = self.model(tokens)[:, -1, :] / temperature
            values, indices = torch.topk(logits, top_k)
            probs = F.softmax(values, dim=-1)
            next_token = indices[0, torch.multinomial(probs, 1)]
            tokens = torch.cat([tokens, next_token.view(1, 1)], dim=1)

        return self.ds.decode(tokens[0].tolist())


# Streamlit Interface
st.set_page_config(page_title="British Chat Model", layout="centered")
st.title("British Chat Model")

@st.cache_resource
def load_system():
    dataset = UKChatDataset()
    model = MiniTransformer(dataset.vocab_size)
    trainer = Trainer(model, dataset)
    trainer.train()
    return Generator(model, dataset)

generator = load_system()

prompt = st.text_input("You:", "Hello how are you today")
if st.button("Reply"):
    response = generator.generate(prompt)
    st.markdown("**Model:**")
    st.write(response)