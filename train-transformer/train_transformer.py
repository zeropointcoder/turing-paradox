import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class TextGenerator:
    def build_corpus(self):
        base = [
            "the colour of the programme was marvellous",
            "she organised the theatre programme carefully",
            "favour and honour are valued in this neighbourhood",
            "the centre of the city has a lovely harbour",
            "travelling requires patience and organisation",
            "the behaviour of the neighbour was admirable"
        ]

        return " ".join(base * 40)


class CharDataset(Dataset):
    def __init__(self, text, seq_len=32):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.data = torch.tensor([self.stoi[c] for c in text])
        self.seq_len = seq_len
        self.vocab_size = len(chars)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + 1:idx + self.seq_len + 1]
        return x, y

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, heads=4, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional = nn.Parameter(torch.zeros(1, 512, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=heads,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x) + self.positional[:, :x.size(1)]
        x = self.encoder(x)
        return self.fc(x)


class Trainer:
    def __init__(self, model, loader):
        self.model = model
        self.loader = loader
        self.loss_fn = nn.CrossEntropyLoss()
        self.opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def train(self, epochs):
        self.model.train()
        print("\n")
        for epoch in range(epochs):
            total_loss = 0.0
            for i, (x, y) in enumerate(self.loader):
                logits = self.model(x)
                loss = self.loss_fn(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                total_loss += loss.item()

                if i % 200 == 0:
                    print(f"epoch {epoch + 1} batch {i} loss {loss.item():.4f}")

            avg = total_loss / len(self.loader)
            print(f"epoch {epoch + 1} completed | avg loss {avg:.4f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    text = TextGenerator().build_corpus()
    dataset = CharDataset(text, seq_len=32)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = TransformerLM(dataset.vocab_size)
    trainer = Trainer(model, loader)
    trainer.train(args.epochs)


if __name__ == "__main__":
    main()