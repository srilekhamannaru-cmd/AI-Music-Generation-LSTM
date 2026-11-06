import numpy as np
import json
import os

print("Loading processed music dataset...")

sequences = np.load("data/sequences.npy")

seed_sequence = sequences[0][:50].tolist()

os.makedirs("data", exist_ok=True)

with open("data/seed_sequence.json", "w") as file:
    json.dump(seed_sequence, file)

print("Seed sequence created successfully!")
print("Total seed notes:", len(seed_sequence))
print("Saved to: data/seed_sequence.json")