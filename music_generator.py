import os
import json
import numpy as np
import torch
import torch.nn as nn
import pretty_midi

# -----------------------------
# Configuration
# -----------------------------

SEQUENCE_LENGTH = 50
NUM_NOTES_TO_GENERATE = 100
TEMPERATURE = 0.8

MODEL_PATH = "models/music_lstm.pth"
MAPPING_PATH = "data/note_mapping.json"
SEQUENCES_PATH = "data/sequences.npy"
OUTPUT_PATH = "outputs/generated_music.mid"

device = torch.device("cpu")


# -----------------------------
# LSTM Model
# -----------------------------

class MusicLSTM(nn.Module):

    def __init__(self, vocab_size):
        super(MusicLSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, 64)

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        self.fc = nn.Linear(128, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output


# -----------------------------
# Load Trained Model
# -----------------------------

print("\nLoading trained music generation model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

vocab_size = checkpoint["vocab_size"]

model = MusicLSTM(vocab_size).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Model loaded successfully!")


# -----------------------------
# Load Note Mapping
# -----------------------------

with open(MAPPING_PATH, "r") as file:
    note_mapping = json.load(file)

int_to_note = {
    index: int(pitch)
    for pitch, index in note_mapping.items()
}


# -----------------------------
# Select Seed Sequence
# -----------------------------

print("\nPreparing seed sequence...")

sequences = np.load(SEQUENCES_PATH)

seed = sequences[0][:SEQUENCE_LENGTH]

generated_sequence = list(seed)

print("Seed sequence prepared!")


# -----------------------------
# Generate Music
# -----------------------------

print("\nGenerating new music...")

with torch.no_grad():

    for _ in range(NUM_NOTES_TO_GENERATE):

        input_sequence = torch.tensor(
            [generated_sequence[-SEQUENCE_LENGTH:]],
            dtype=torch.long
        ).to(device)

        predictions = model(input_sequence)

        probabilities = torch.softmax(
            predictions / TEMPERATURE,
            dim=1
        )

        next_note = torch.multinomial(
            probabilities,
            num_samples=1
        ).item()

        generated_sequence.append(next_note)


print("Music generation completed!")


# -----------------------------
# Convert Notes into MIDI
# -----------------------------

print("\nConverting generated notes into MIDI...")

midi = pretty_midi.PrettyMIDI()

instrument = pretty_midi.Instrument(program=0)

start_time = 0.0
note_duration = 0.35
note_interval = 0.4

generated_notes = generated_sequence[SEQUENCE_LENGTH:]

for note_id in generated_notes:

    pitch = int_to_note[note_id]

    note = pretty_midi.Note(
        velocity=100,
        pitch=pitch,
        start=start_time,
        end=start_time + note_duration
    )

    instrument.notes.append(note)

    start_time += note_interval


midi.instruments.append(instrument)


# -----------------------------
# Save Generated Music
# -----------------------------

os.makedirs("outputs", exist_ok=True)

midi.write(OUTPUT_PATH)

print("\n===================================")
print("AI MUSIC GENERATED SUCCESSFULLY!")
print("===================================")
print(f"Total generated notes: {len(generated_notes)}")
print(f"Saved MIDI file: {OUTPUT_PATH}")
print("===================================")