import os
import glob
import json
import numpy as np
from music21 import converter, note, chord

# Project folders
MIDI_FOLDER = "data/midi"
OUTPUT_FOLDER = "data"

# LSTM sequence length
SEQUENCE_LENGTH = 50


def extract_notes(file_path):
    """Extract notes and chords from a MIDI file."""

    try:
        midi = converter.parse(file_path)
        notes = []

        for element in midi.flatten().notes:
            if isinstance(element, note.Note):
                notes.append(element.pitch.midi)

            elif isinstance(element, chord.Chord):
                for pitch in element.pitches:
                    notes.append(pitch.midi)

        return notes

    except Exception as error:
        print(f"Skipping {os.path.basename(file_path)}: {error}")
        return []


def preprocess_dataset():

    print("\nStarting MIDI preprocessing...\n")

    midi_files = glob.glob(os.path.join(MIDI_FOLDER, "*.mid"))
    midi_files += glob.glob(os.path.join(MIDI_FOLDER, "*.midi"))

    print(f"Found {len(midi_files)} MIDI files.")

    all_notes = []

    for index, file_path in enumerate(midi_files, start=1):

        notes = extract_notes(file_path)

        if notes:
            all_notes.extend(notes)

        if index % 50 == 0:
            print(f"Processed {index}/{len(midi_files)} files.")

    if len(all_notes) <= SEQUENCE_LENGTH:
        print("Not enough musical notes to create training sequences.")
        return

    # Create a vocabulary of unique pitches
    unique_notes = sorted(set(all_notes))

    note_to_int = {
        pitch: index
        for index, pitch in enumerate(unique_notes)
    }

    # Convert pitches into numerical values
    encoded_notes = [
        note_to_int[pitch]
        for pitch in all_notes
    ]

    # Create input sequences and targets
    sequences = []

    for i in range(len(encoded_notes) - SEQUENCE_LENGTH):
        sequence = encoded_notes[i:i + SEQUENCE_LENGTH]
        target = encoded_notes[i + SEQUENCE_LENGTH]

        sequences.append(sequence + [target])

    sequences = np.array(sequences, dtype=np.int32)

    # Save processed data
    np.save(
        os.path.join(OUTPUT_FOLDER, "sequences.npy"),
        sequences
    )

    with open(
        os.path.join(OUTPUT_FOLDER, "note_mapping.json"),
        "w"
    ) as file:
        json.dump(note_to_int, file)

    print("\nPreprocessing completed successfully!")
    print(f"Total notes extracted: {len(all_notes)}")
    print(f"Unique musical pitches: {len(unique_notes)}")
    print(f"Training sequences created: {len(sequences)}")
    print("Saved: data/sequences.npy")
    print("Saved: data/note_mapping.json")


if __name__ == "__main__":
    preprocess_dataset()