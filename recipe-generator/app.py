import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random


class RecipeDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.targets[idx])


class RecipeLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=50, hidden_size=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x[:, -1, :])
        return x, hidden


class RecipeGeneratorAI:
    def __init__(self, corpus, seq_length=5, epochs=50):
        self.corpus = corpus
        self.seq_length = seq_length
        self.epochs = epochs
        self.word2idx = {}
        self.idx2word = {}
        self.model = None
        self._prepare_data()
        self._build_model()
        self._train_model()

    def _prepare_data(self):
        words = set(word.lower() for line in self.corpus for word in line.split())
        self.word2idx = {w: i for i, w in enumerate(words)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        vocab_size = len(words)
        self.vocab_size = vocab_size

        sequences, targets = [], []
        for line in self.corpus:
            tokens = [self.word2idx[w.lower()] for w in line.split()]
            for i in range(len(tokens) - self.seq_length):
                sequences.append(tokens[i:i+self.seq_length])
                targets.append(tokens[i+self.seq_length])

        self.sequences = sequences
        self.targets = targets

    def _build_model(self):
        self.model = RecipeLSTM(self.vocab_size)

    def _train_model(self):
        dataset = RecipeDataset(self.sequences, self.targets)
        loader = DataLoader(dataset, batch_size=16, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        self.model.train()
        for _ in range(self.epochs):
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs, _ = self.model(x_batch)
                loss = loss_fn(outputs, y_batch)
                loss.backward()
                optimizer.step()

    def generate_recipe(self, start_word=None, length=50):
        if start_word is None or start_word.lower() not in self.word2idx:
            start_word = random.choice(list(self.word2idx.keys()))
        current_sequence = [self.word2idx[start_word.lower()]] * self.seq_length
        recipe = [start_word.capitalize()]
        self.model.eval()
        hidden = None

        for _ in range(length-1):
            x = torch.tensor([current_sequence])
            with torch.no_grad():
                out, hidden = self.model(x, hidden)
                probs = F.softmax(out, dim=1).numpy().flatten()
                next_idx = np.random.choice(len(probs), p=probs)
                next_word = self.idx2word[next_idx]
            recipe.append(next_word)
            current_sequence = current_sequence[1:] + [next_idx]

        return ' '.join(recipe)


CORPUS = [
    "Preheat the oven to 180 degrees Celsius.",
    "Grease a baking pan with butter.",
    "Mix flour and sugar in a large bowl.",
    "Crack the eggs and whisk until smooth.",
    "Add milk gradually to the mixture.",
    "Stir in vanilla extract for flavor.",
    "Fold in chocolate chips carefully.",
    "Pour the batter into the prepared pan.",
    "Bake for 25 minutes or until golden brown.",
    "Let the cake cool before slicing.",
    "Chop onions finely for the recipe.",
    "Sauté onions in olive oil until soft.",
    "Add garlic and cook for two minutes.",
    "Dice tomatoes and add to the pan.",
    "Season with salt and black pepper.",
    "Simmer the sauce for 15 minutes.",
    "Boil pasta in salted water until tender.",
    "Drain the pasta and return to the pot.",
    "Mix the pasta with the tomato sauce.",
    "Grate cheese over the pasta before serving.",
    "Season chicken with paprika and salt.",
    "Heat oil in a pan and fry the chicken.",
    "Cook chicken until golden and crisp.",
    "Chop bell peppers into small pieces.",
    "Stir-fry bell peppers with onions.",
    "Add soy sauce and mix well.",
    "Marinate prawns with garlic and soy sauce.",
    "Grill the prawns for five minutes each side.",
    "Slice prawns thinly and serve hot.",
    "Wash and chop fresh lettuce leaves.",
    "Prepare a dressing with olive oil and lemon.",
    "Toss salad with the dressing gently.",
    "Crack eggs into a bowl and beat lightly.",
    "Cook eggs in a non-stick pan on medium heat.",
    "Add salt and herbs while cooking eggs.",
    "Toast bread slices until golden brown.",
    "Spread butter on the toasted bread.",
    "Slice avocado and place on toast.",
    "Sprinkle salt and chili flakes on top.",
    "Peel carrots and cut into sticks.",
    "Steam carrots until tender but firm.",
    "Boil potatoes until soft for mashing.",
    "Mash potatoes with milk and butter.",
    "Chop fresh parsley to garnish dishes.",
    "Heat oil in a wok before stir-frying.",
    "Add tofu cubes and cook until golden.",
    "Combine rice and vegetables in a pan.",
    "Season rice with soy sauce and sesame oil.",
    "Serve meals hot with fresh herbs.",
    "Preheat grill before cooking vegetables."
]


st.title("AI-Powered Recipe Generator (PyTorch)")
st.markdown("Generate creative recipes using a lightweight neural network trained on cooking instructions.")

generator = RecipeGeneratorAI(CORPUS, epochs=30)

ingredient_input = st.text_input("Enter a starting ingredient or word (optional):")
length_input = st.slider("Recipe length (words)", min_value=20, max_value=100, value=50)

if st.button("Generate Recipe"):
    recipe = generator.generate_recipe(start_word=ingredient_input, length=length_input)
    st.text_area("Generated Recipe", value=recipe, height=300)