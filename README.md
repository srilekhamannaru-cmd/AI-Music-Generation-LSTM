# 🎵 AI Music Generation Using LSTM

## 🚀 Live Demo

Try the AI Music Generator here:

🎵 **[Launch AI Music Generation App](https://ai-music-generation-lstm.streamlit.app/)**

Generate original music using an LSTM deep learning model and listen to the results directly in your browser.

## About the Project

AI Music Generation is a deep learning project that generates new musical sequences using a Long Short-Term Memory (LSTM) neural network.

The model learns musical pitch patterns from MIDI files and uses the learned patterns to generate new sequences. These sequences are converted into MIDI and WAV audio files.

The project also includes a Streamlit web application where users can generate music, listen to the output and download their files.

## ✨ Features

- MIDI dataset preprocessing
- Musical note and pitch extraction
- LSTM-based deep learning model
- AI-generated musical sequences
- MIDI file generation
- WAV audio conversion
- Interactive Streamlit interface
- Audio playback
- MIDI and WAV downloads

## 🛠️ Technologies Used

- Python
- PyTorch
- NumPy
- Music21
- PrettyMIDI
- Streamlit
- LSTM Neural Networks

## 📂 Project Structure

```text
CodeAlpha_MusicGeneration/
│
├── data/
│   ├── midi/
│   ├── sequences.npy
│   └── note_mapping.json
│
├── models/
│   └── music_lstm.pth
│
├── outputs/
│   ├── generated_music.mid
│   └── generated_music.wav
│
├── preprocess.py
├── train.py
├── music_generator.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd CodeAlpha_MusicGeneration
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the MIDI Dataset

Download the Public Piano MIDI Dataset and place the MIDI files inside:

```text
data/midi/
```

Follow the dataset's CC BY-NC-SA 4.0 license and attribution requirements.

### 5. Preprocess the Dataset

```bash
python preprocess.py
```

This extracts musical pitches and creates training sequences.

### 6. Train the LSTM Model

```bash
python train.py
```

The trained model will be saved inside:

```text
models/music_lstm.pth
```

### 7. Generate Music

```bash
python music_generator.py
```

Generated MIDI files will be saved inside the outputs folder.

### 8. Run the Streamlit Application

```bash
streamlit run app.py
```

Open the local application in your browser:

```text
http://localhost:8501
```

## 🧠 Model Architecture

The project uses an LSTM-based neural network containing:

- Embedding Layer
- Two LSTM Layers
- Fully Connected Output Layer

The model learns patterns from sequences of musical pitches and predicts the next note.

## 🎧 Generated Output

The application supports:

- MIDI music generation
- WAV audio conversion
- In-browser audio playback
- MIDI file download
- WAV audio download

## 📌 Current Implementation

The current model was trained for five epochs using a subset of 100,000 training sequences extracted from 500 MIDI files.

It generates pitch sequences with fixed note durations. The WAV preview uses synthesized tones rather than a realistic instrument soundfont.

## 🚀 Future Improvements

- Generate more realistic musical compositions.
- Introduce rhythm and note-duration prediction.
- Experiment with larger datasets.
- Improve audio quality using instrument soundfonts.
- Explore more advanced deep learning architectures.

## 👩‍💻 Author

**Sri Lekha Mannaru**

B.Tech Student | Computer Science

---

Built with Python, PyTorch and Streamlit.