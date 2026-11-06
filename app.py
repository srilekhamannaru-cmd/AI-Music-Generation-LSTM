import os
import json
import wave
import numpy as np
import torch
import torch.nn as nn
import pretty_midi
import streamlit as st

# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
    layout="centered"
)

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
        return self.fc(output[:, -1, :])


# -----------------------------
# Load Trained Model
# -----------------------------

@st.cache_resource
def load_music_model():

    device = torch.device("cpu")

    checkpoint = torch.load(
        "models/music_lstm.pth",
        map_location=device
    )

    vocab_size = checkpoint["vocab_size"]

    model = MusicLSTM(vocab_size)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    with open("data/note_mapping.json", "r") as file:
        note_mapping = json.load(file)

    int_to_note = {
        index: int(pitch)
        for pitch, index in note_mapping.items()
    }

    sequences = np.load("data/sequences.npy")

    return model, int_to_note, sequences


# -----------------------------
# Generate Music
# -----------------------------

def generate_music(
    model,
    int_to_note,
    sequences,
    num_notes,
    temperature
):

    sequence_length = 50

    seed = sequences[0][:sequence_length]

    generated_sequence = list(seed)

    with torch.no_grad():

        for _ in range(num_notes):

            input_sequence = torch.tensor(
                [generated_sequence[-sequence_length:]],
                dtype=torch.long
            )

            predictions = model(input_sequence)

            probabilities = torch.softmax(
                predictions / temperature,
                dim=1
            )

            next_note = torch.multinomial(
                probabilities,
                num_samples=1
            ).item()

            generated_sequence.append(next_note)

    # Convert notes into MIDI

    midi = pretty_midi.PrettyMIDI()

    instrument = pretty_midi.Instrument(program=0)

    start_time = 0.0
    note_duration = 0.35
    note_interval = 0.4

    generated_notes = generated_sequence[sequence_length:]

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

    os.makedirs("outputs", exist_ok=True)

    midi_path = "outputs/generated_music.mid"

    midi.write(midi_path)

    return midi_path


# -----------------------------
# Convert MIDI Notes to WAV
# -----------------------------

def create_audio(midi_path):

    midi = pretty_midi.PrettyMIDI(midi_path)

    sample_rate = 22050

    duration = midi.get_end_time() + 0.5

    audio = np.zeros(
        int(sample_rate * duration),
        dtype=np.float32
    )

    for instrument in midi.instruments:

        for note in instrument.notes:

            start = int(note.start * sample_rate)
            end = int(note.end * sample_rate)

            length = end - start

            if length <= 0:
                continue

            t = np.arange(length) / sample_rate

            frequency = pretty_midi.note_number_to_hz(
                note.pitch
            )

            # Create a simple musical tone

            tone = (
                np.sin(2 * np.pi * frequency * t)
                + 0.3 * np.sin(4 * np.pi * frequency * t)
                + 0.1 * np.sin(6 * np.pi * frequency * t)
            )

            # Apply a smooth note envelope

            envelope = np.exp(-4 * t)

            sound = tone * envelope

            end_position = min(end, len(audio))

            audio[start:end_position] += sound[:end_position-start]

    # Normalize audio

    maximum = np.max(np.abs(audio))

    if maximum > 0:
        audio = audio / maximum

    audio = (audio * 32767).astype(np.int16)

    wav_path = "outputs/generated_music.wav"

    # Save WAV file

    with wave.open(wav_path, "wb") as wav_file:

        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        wav_file.writeframes(audio.tobytes())

    return wav_path


# -----------------------------
# Streamlit Interface
# -----------------------------

st.title("🎵 AI Music Generator")

st.write(
    "Create original musical sequences using "
    "a deep learning model trained on MIDI music."
)

st.divider()

st.subheader("🎹 Generate Your Music")

num_notes = st.slider(
    "Select the number of notes",
    min_value=50,
    max_value=300,
    value=100,
    step=50
)

temperature = st.slider(
    "Music creativity",
    min_value=0.5,
    max_value=1.5,
    value=0.8,
    step=0.1
)

st.write(f"Selected notes: **{num_notes}**")

if st.button("🎼 Generate Music", use_container_width=True):

    try:

        with st.spinner("Generating your music..."):

            model, int_to_note, sequences = load_music_model()

            midi_path = generate_music(
                model,
                int_to_note,
                sequences,
                num_notes,
                temperature
            )

            wav_path = create_audio(midi_path)

        st.success("Your AI-generated music is ready! 🎉")

        st.subheader("🎧 Listen to Your Music")

        with open(wav_path, "rb") as audio_file:
            audio_data = audio_file.read()

        st.audio(
            audio_data,
            format="audio/wav"
        )

        st.subheader("⬇️ Download Your Music")

        with open(midi_path, "rb") as midi_file:
            midi_data = midi_file.read()

        st.download_button(
            "Download MIDI",
            data=midi_data,
            file_name="generated_music.mid",
            mime="audio/midi",
            use_container_width=True
        )

        st.download_button(
            "Download WAV Audio",
            data=audio_data,
            file_name="generated_music.wav",
            mime="audio/wav",
            use_container_width=True
        )

        st.info(
            f"Successfully generated {num_notes} musical notes!"
        )

    except Exception as error:

        st.error(f"Something went wrong: {error}")

st.divider()

st.caption(
    "Built using Python, PyTorch, LSTM, NumPy and PrettyMIDI."
)