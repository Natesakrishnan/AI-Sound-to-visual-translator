# AI-Based Sound-to-Visual Translation for Intelligent Audio Pattern Recognition

An end-to-end intelligent desktop application that processes environmental audio signals, extracts multi-domain physical audio features, performs deep convolutional neural classification on the **ESC-50 dataset**, and converts the sound characteristics into deterministic, generative visual artwork.

---

## 📌 Project Overview & Key Innovation

Most conventional audio AI systems only output a text category label. This project pioneers a **deterministic sound-to-visual translation engine** that maps the actual acoustic physics of the input audio into beautiful geometric patterns.

```text
Audio File (.wav)
       ↓
Audio Preprocessing (Mono, 22.05 kHz, Peak Normalization)
       ↓
Multi-Feature Extraction (FFT, MFCC, Mel Spectrogram, Centroid, ZCR, RMS, Chroma)
       ↓
CNN-Based Audio Classification (50 ESC-50 Classes)
       ↓
Predicted Class + Confidence Score
       ↓
Acoustic Feature Analysis & Mathematical Parameterization
       ↓
Sound-to-Visual Translation Engine
       ↓
Generated Visual Art & High-Resolution PNG Export
```

> **Important Distinction**: The final generated visual is NOT simply a Mel Spectrogram plot. The spectrogram is only an intermediate feature matrix for the CNN; the visual translation is a separate, mathematical mapping from audio physics into generative sacred geometry and particle systems.

---

## 🚀 Key Features

- **Tkinter Desktop GUI**: Sleek dark-mode desktop interface with 5 dedicated functional tabs.
- **Signal Analysis Visualizer**: Live 4-grid visualizer rendering Waveform $y(t)$, FFT Spectrum, 20-band MFCC matrix, and Log-Mel Spectrogram.
- **Embedded Audio Player**: Play, pause, and stop controls powered by `pygame.mixer`.
- **4-Block 2D CNN**: Convolutional Neural Network built with **Keras 3 (PyTorch Backend)**, optimized for CPU and NVIDIA CUDA GPU execution.
- **ESC-50 Benchmark Compliance**: Strict fold partitioning:
  - **Folds 1, 2, 3** $\rightarrow$ Training Set (1,200 audio files)
  - **Fold 4** $\rightarrow$ Validation Set (400 audio files)
  - **Fold 5** $\rightarrow$ Final Unseen Test Set (400 audio files)
- **Lightning-Fast Feature Caching**: Precomputed Mel Spectrograms cached to disk (`esc50_features_cache.npz`), avoiding repeated extraction.
- **5 Generative Visual Modes**:
  1. *Class-Specific Generative Geometry* (Mapped to 50 verified ESC-50 classes)
  2. *Sound Mandala / Energy Aura*
  3. *Frequency Harmonic Spiral*
  4. *Waveform Geometric Flower*
  5. *Acoustic Particle Constellation*
- **Mathematical Telemetry**: Live readout of exact parameters (Radius, Rotation Angle, Vertex Count, Harmonic Layers, Color Palette).
- **Comprehensive Evaluation**: High-resolution 50-Class Confusion Matrix heatmap, Training Loss/Accuracy curves, Accuracy, Precision, Recall, and F1-score.

---

## 🛠 Technologies & Frameworks

| Component | Technology |
| :--- | :--- |
| **Language & GUI** | Python 3.10+, Tkinter, TTK, Pillow |
| **Deep Learning** | Keras 3 with PyTorch Backend (`torch`, `keras`) |
| **Audio Processing** | LibROSA, SoundFile, SciPy FFT |
| **Data & Math** | NumPy, Pandas, Scikit-learn |
| **Visualization** | Matplotlib (Embedded FigureCanvasTkAgg) |
| **Audio Playback** | PyGame Mixer |

---

## 📂 Project Folder Structure

```text
AI_Sound_To_Visual_Translator/
│
├── main.py                     # Main application entry point & startup diagnostics
├── requirements.txt            # Complete Python dependencies
├── README.md                   # Comprehensive project documentation
├── run.bat                     # Windows batch launcher for desktop GUI
├── train_model.bat             # Windows batch script for standalone model training
│
├── dataset/                    # ESC-50 dataset directory
│   ├── audio/                  # 2,000 .wav files
│   └── esc50.csv               # Dataset metadata & fold definitions
│
├── models/
│   ├── audio_cnn_model.keras   # Saved trained Keras 3 CNN model
│   ├── label_mapping.json      # 50-class index to category mapping
│   └── model_metadata.json     # Training execution stats, device info, test metrics
│
├── src/
│   ├── __init__.py
│   ├── hardware_info.py        # CPU, RAM, discrete GPU, VRAM, and CUDA detector
│   ├── audio_preprocessing.py  # Audio loader, resampling, normalization, audio player
│   ├── feature_extraction.py   # FFT, MFCC, Mel Spectrogram, Centroid, ZCR, RMS, Chroma
│   ├── dataset_loader.py       # ESC-50 parser, fold-based splitter, feature caching
│   ├── cnn_model.py            # 4-block 2D CNN architecture in Keras 3
│   ├── train_model.py          # Standalone training script with early stopping & curves
│   ├── prediction.py           # AudioPredictor class with Top-5 confidence rankings
│   ├── visual_translation.py   # Deterministic sound-to-visual generative translation engine
│   ├── evaluation.py           # Precision, Recall, F1, 50-Class Confusion Matrix plotter
│   └── gui.py                  # 5-tab desktop GUI application with embedded canvases
│
├── outputs/
│   ├── cache/                  # Compressed NumPy feature cache
│   ├── graphs/                 # Loss & Accuracy training curves
│   ├── confusion_matrix/       # 50-Class Confusion Matrix & test metrics JSON
│   ├── spectrograms/           # Exported spectrogram plots
│   ├── predictions/            # Prediction logs
│   └── generated_visuals/      # Exported Sound-to-Visual PNG artwork
│
└── docs/
    ├── installation.md         # Step-by-step Windows installation guide
    ├── execution.md            # Execution workflow & tab guide
    └── project_architecture.md # Viva-friendly technical Q&A and architecture
```

---

## 🔬 Mathematical Sound-to-Visual Mappings

The translation engine converts acoustic signal properties into visual geometry through deterministic mathematical relationships:

1. **RMS Energy $\rightarrow$ Visual Radius & Scale**:
   $$R_{\text{base}} = 1.0 + 1.4 \times \text{RMS}_{\text{norm}}$$
2. **Spectral Centroid $\rightarrow$ Angular Rotation**:
   $$\theta_{\text{rot}} = 2\pi \times \text{Centroid}_{\text{norm}}$$
3. **Zero-Crossing Rate (ZCR) $\rightarrow$ Vertex Count & Edge Complexity**:
   $$N_{\text{vertices}} = 4 + \lfloor 28 \times \text{ZCR}_{\text{norm}} \rfloor$$
4. **Dominant Frequency $\rightarrow$ Harmonic Layer Count**:
   $$N_{\text{layers}} = 2 + \lfloor 6 \times \text{Freq}_{\text{norm}} \rfloor$$
5. **Chroma Pitch Class $\rightarrow$ Chromatic Palette**:
   $$\text{Hue}_{\text{primary}} = \frac{\text{PitchIndex} \times 30^\circ}{360^\circ}$$
6. **Class Archetypes for 50 ESC-50 Classes**:
   - **Animals** (Dog, Rooster, Pig, Cow, Frog, Cat, Hen, Insects, Sheep, Crow) $\rightarrow$ Organic Concentric Orbital Auras
   - **Nature / Water** (Rain, Sea Waves, Fire, Birds, Water Drops, Wind, Thunderstorm) $\rightarrow$ Flowing Streamline Fluid Cascades
   - **Human Sounds** (Crying Baby, Sneezing, Clapping, Breathing, Coughing, Footsteps, Laughing) $\rightarrow$ Vocal Tract Standing Wave Nodes
   - **Interior / Domestic** (Door Creaks, Mouse Click, Keyboard, Knock, Clock, Glass Breaking) $\rightarrow$ Mechanical Geometric Lattice Grids
   - **Urban / Mechanical** (Helicopter, Chainsaw, Siren, Car Horn, Engine, Train, Airplane, Fireworks) $\rightarrow$ Crystalline Angular Polygonal Bursts

---

## ⚡ Quick Start Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the CNN Model
```bash
python src/train_model.py
```
*(Or double click `train_model.bat`)*

### 3. Launch the Desktop Application
```bash
python main.py
```
*(Or double click `run.bat`)*

---

## 📊 Evaluation & Verification

- The model is evaluated on strictly unseen **Fold 5 test audio files** (400 samples across 50 balanced classes).
- Metrics calculated: Overall Accuracy, Macro-Averaged Precision, Macro-Averaged Recall, and Macro-Averaged F1-Score.
- A high-resolution 50-class confusion matrix is generated and saved under `outputs/confusion_matrix/confusion_matrix.png`.
