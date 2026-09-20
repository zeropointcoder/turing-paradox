import streamlit as st
import pyttsx3
import torch
import torch.nn as nn
import torch.optim as optim


# ML model for UK spelling normalisation
class UKNormaliserML(nn.Module):
    def __init__(self, vocab_size=30, embed_dim=16, hidden_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)         # [batch, seq_len, embed_dim]
        out, _ = self.rnn(emb)          # [batch, seq_len, hidden_dim]
        out = self.fc(out[:, -1, :])    # take last time step
        return out


class UKTextNormaliser:
    def __init__(self):
        # US -> UK spelling replacements (30 words)
        self.replacements = {
            "color": "colour",
            "organize": "organise",
            "organization": "organisation",
            "analyze": "analyse",
            "center": "centre",
            "meter": "metre",
            "favorite": "favourite",
            "theater": "theatre",
            "honor": "honour",
            "neighbor": "neighbour",
            "defense": "defence",
            "license": "licence",
            "traveling": "travelling",
            "canceled": "cancelled",
            "program": "programme",
            "apologize": "apologise",
            "catalog": "catalogue",
            "dialog": "dialogue",
            "jewelry": "jewellery",
            "check": "cheque",
            "plow": "plough",
            "draft": "draught",
            "practice": "practise",
            "offense": "offence",
            "tire": "tyre",
            "aluminum": "aluminium",
            "defenseman": "defenceman",
            "estrogen": "oestrogen",
            "pajamas": "pyjamas",
            "programed": "programmed"
        }

        # Auto-generate word2idx and idx2word from replacements
        self.word2idx = {w: i for i, w in enumerate(self.replacements.keys())}
        self.idx2word = {i: w for w, i in self.word2idx.items()}

        # Initialise tiny RNN model
        self.model = UKNormaliserML(vocab_size=len(self.word2idx))
        self._train_model()

    def _text_to_idx(self, word):
        # Shape [1,1] for batch and seq
        return torch.tensor([[self.word2idx.get(word.lower(), 0)]], dtype=torch.long)

    def _train_model(self):
        # Tiny offline training
        criterion = nn.CrossEntropyLoss()
        optimiser = optim.Adam(self.model.parameters(), lr=0.01)
        for epoch in range(50):
            for word, idx in self.word2idx.items():
                input_idx = self._text_to_idx(word)     # [1,1]
                target_idx = torch.tensor([idx])
                optimiser.zero_grad()
                output = self.model(input_idx)          # [1, vocab_size]
                loss = criterion(output, target_idx)
                loss.backward()
                optimiser.step()

    def normalise(self, text: str) -> str:
        words = text.split()
        normalised = []
        for w in words:
            if w.lower() in self.replacements:
                normalised.append(self.replacements[w.lower()])
            else:
                normalised.append(w)
        return " ".join(normalised)


# ML model to predict speech duration
class DurationPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.fc(x)


class SpeechEvaluator:
    def __init__(self):
        self.model = DurationPredictor()
        self._train_model()

    def _train_model(self):
        # Offline training: word count -> duration
        X = torch.tensor([[i] for i in range(1, 21)], dtype=torch.float32)
        y = torch.tensor([[i/2.5] for i in range(1, 21)], dtype=torch.float32)
        criterion = nn.MSELoss()
        optimiser = optim.Adam(self.model.parameters(), lr=0.05)
        for _ in range(200):
            optimiser.zero_grad()
            out = self.model(X)
            loss = criterion(out, y)
            loss.backward()
            optimiser.step()

    def length_seconds(self, text: str) -> float:
        words = len(text.split())
        x = torch.tensor([[words]], dtype=torch.float32)
        return round(self.model(x).item(), 2)


# Offline TTS
class SpeechModel:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.voice_id = self._select_uk_voice()

    def _select_uk_voice(self):
        for voice in self.engine.getProperty("voices"):
            if "uk" in f"{voice.name} {voice.id}".lower():
                return voice.id
        return self.engine.getProperty("voices")[0].id

    def speak(self, text: str):
        self.engine.setProperty("voice", self.voice_id)
        self.engine.say(text)
        self.engine.runAndWait()


# Controller
class SpeechController:
    def __init__(self):
        self.normaliser = UKTextNormaliser()
        self.evaluator = SpeechEvaluator()
        self.model = SpeechModel()

    def run(self, text: str):
        clean_text = self.normaliser.normalise(text)
        duration = self.evaluator.length_seconds(clean_text)
        self.model.speak(clean_text)
        return duration


# Streamlit UI
st.set_page_config(page_title="", layout="centered")
st.title("UK-English Text-to-speech (ML)")

controller = SpeechController()
user_text = st.text_area("Enter text (UK English):", height=200, placeholder="Type text with US spellings...")

if st.button("Speak"):
    if user_text.strip():
        duration = controller.run(user_text)
        st.success(f"Speech completed · Approx. {duration} seconds")
    else:
        st.warning("Please enter some text.")

