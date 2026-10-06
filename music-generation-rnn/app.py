import streamlit as st
import numpy as np
from mido import Message, MidiFile, MidiTrack


# Synthetic dataset
class MusicDataset:
    def __init__(self, sequence_length=10, vocab_size=50, num_sequences=500):
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.num_sequences = num_sequences
        self.data = self.generate_data()

    def generate_data(self):
        return np.random.randint(0, self.vocab_size, size=(self.num_sequences, self.sequence_length))

    def get_train_data(self):
        X = self.data[:, :-1]
        y = self.data[:, -1]
        return X, y


# Simple NumPy RNN
class SimpleRNN:
    def __init__(self, vocab_size, hidden_size=32):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def train(self, X, y, epochs=5, lr=0.1):
        for epoch in range(epochs):
            loss = 0
            for i in range(len(X)):
                x_seq = X[i]
                target = y[i]
                hs = np.zeros((self.hidden_size, 1))
                # Forward pass
                for t in x_seq:
                    x_vec = np.zeros((self.vocab_size, 1))
                    x_vec[t] = 1
                    hs = np.tanh(np.dot(self.Wxh, x_vec) + np.dot(self.Whh, hs) + self.bh)
                y_pred = self.softmax(np.dot(self.Why, hs) + self.by)
                loss -= np.log(y_pred[target,0] + 1e-9)
                # Simplified backprop on output layer
                dy = y_pred
                dy[target] -= 1
                self.Why -= lr * np.dot(dy, hs.T)
                self.by -= lr * dy
            st.write(f"Epoch {epoch+1}/{epochs} Loss: {loss/len(X):.4f}")

    def generate_sequence(self, seed, length=20):
        hs = np.zeros((self.hidden_size,1))
        sequence = list(seed)
        for t in seed:
            x_vec = np.zeros((self.vocab_size,1))
            x_vec[t] = 1
            hs = np.tanh(np.dot(self.Wxh, x_vec) + np.dot(self.Whh, hs) + self.bh)
        for _ in range(length):
            y_pred = self.softmax(np.dot(self.Why, hs) + self.by)
            next_note = np.argmax(y_pred)
            sequence.append(next_note)
            x_vec = np.zeros((self.vocab_size,1))
            x_vec[next_note] = 1
            hs = np.tanh(np.dot(self.Wxh, x_vec) + np.dot(self.Whh, hs) + self.bh)
        return sequence


# Convert integer sequence to MIDI
def sequence_to_midi(sequence, filename="melody.mid", base_note=60):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    for note in sequence:
        midi_note = base_note + note  # Map 0-49 to MIDI notes 60-109
        track.append(Message('note_on', note=midi_note, velocity=64, time=200))
        track.append(Message('note_off', note=midi_note, velocity=64, time=200))
    mid.save(filename)
    return filename


# Streamlit interface
st.title("Melody Synthesiser (Playable MIDI)")

sequence_length = st.slider("Sequence length", 5, 20, 10)
num_sequences = st.slider("Number of sequences", 100, 1000, 500)

dataset = MusicDataset(sequence_length=sequence_length, num_sequences=num_sequences)
X, y = dataset.get_train_data()
st.write("Dataset generated with shape:", X.shape)

rnn = SimpleRNN(vocab_size=dataset.vocab_size)

if st.button("Train Model"):
    rnn.train(X, y, epochs=5)
    st.success("Model trained!")

seed_input = st.text_input("Enter seed notes (comma separated, e.g., 1,5,3):", "1,2,3")
if st.button("Generate Playable Melody"):
    try:
        seed_notes = [int(i.strip()) for i in seed_input.split(",")]
        melody = rnn.generate_sequence(seed_notes, length=20)
        st.write("Generated melody sequence:", melody)
        midi_file = sequence_to_midi(melody, filename="melody.mid")
        with open(midi_file, "rb") as f:
            st.download_button("Download MIDI", f, file_name="melody.mid")
    except Exception as e:
        st.error("Invalid input. Please enter integers separated by commas.")