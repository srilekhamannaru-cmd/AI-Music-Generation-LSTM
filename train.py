import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Configuration
DATA_PATH = "data/sequences.npy"
MAPPING_PATH = "data/note_mapping.json"
MODEL_PATH = "models/music_lstm.pth"

SEQUENCE_LENGTH = 50
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001
TRAINING_SAMPLES = 100000

# Use CPU for training
device = torch.device("cpu")


# LSTM Music Generation Model
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

        output = output[:, -1, :]

        output = self.fc(output)

        return output


def train_model():

    print("\nLoading processed music dataset...")

    sequences = np.load(DATA_PATH)

    with open(MAPPING_PATH, "r") as file:
        note_mapping = json.load(file)

    vocab_size = len(note_mapping)

    print("Dataset shape:", sequences.shape)
    print("Vocabulary size:", vocab_size)

    # Randomly select a manageable training subset
    rng = np.random.default_rng(42)

    sample_count = min(TRAINING_SAMPLES, len(sequences))

    indices = rng.choice(
        len(sequences),
        size=sample_count,
        replace=False
    )

    selected_data = sequences[indices]

    # Separate input notes and target notes
    X = torch.tensor(
        selected_data[:, :-1],
        dtype=torch.long
    )

    y = torch.tensor(
        selected_data[:, -1],
        dtype=torch.long
    )

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # Initialize model
    model = MusicLSTM(vocab_size).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print("\nStarting LSTM training...\n")

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            predictions = model(batch_X)

            loss = criterion(predictions, batch_y)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(loader)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"- Loss: {average_loss:.4f}"
        )

    # Save trained model
    os.makedirs("models", exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "sequence_length": SEQUENCE_LENGTH
        },
        MODEL_PATH
    )

    print("\nTraining completed successfully!")
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    train_model()