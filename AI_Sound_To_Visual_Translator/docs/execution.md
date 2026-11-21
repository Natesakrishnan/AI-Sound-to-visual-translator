# Execution Guide

## AI-Based Sound-to-Visual Translation for Intelligent Audio Pattern Recognition

This guide explains how to train the deep learning model, evaluate performance, and run the desktop GUI application.

---

### Step 1: Verify Dataset Placement

The application automatically looks for the ESC-50 dataset in `K:/AI project/Datasets` or relative `../Datasets` or `dataset/`. Ensure the directory contains:

```text
Datasets/
├── audio/            (Contains 2,000 .wav files)
└── esc50.csv         (Contains metadata columns: filename, fold, target, category, esc10, src_file, take)
```

---

### Step 2: Model Training & Evaluation

To train the CNN model from the command line:

```bash
python src/train_model.py
```

*Or double-click `train_model.bat` in Windows Explorer.*

#### Training Workflow:
1. Detects system hardware (CPU, RAM, GPU).
2. Extracts & caches 2,000 Mel Spectrograms into `outputs/cache/esc50_features_cache.npz` (subsequent training runs start instantly).
3. Partitions data:
   - **Folds 1, 2, 3**: Training (1,200 audio files)
   - **Fold 4**: Validation (400 audio files)
   - **Fold 5**: Unseen Test (400 audio files)
4. Trains the 4-block Conv2D network with early stopping and learning rate scheduling.
5. Evaluates on Fold 5 and generates:
   - `models/audio_cnn_model.keras`
   - `models/label_mapping.json`
   - `models/model_metadata.json`
   - `outputs/graphs/training_history.png`
   - `outputs/confusion_matrix/confusion_matrix.png`
   - `outputs/confusion_matrix/test_metrics.json`

---

### Step 3: Launching the Desktop GUI Application

To launch the full graphical desktop application:

```bash
python main.py
```

*Or double-click `run.bat` in Windows Explorer.*

---

### Step 4: Using the Application Tabs

#### 1. Home Tab
- Inspect hardware detection details (CPU cores, RAM, GPU, DL device).
- Check CNN model readiness badge.
- Click quick action buttons for common tasks.

#### 2. Audio Analysis Tab
- Click **"Select WAV File"** and choose any sound file.
- Inspect acoustic metadata (duration, sample rate, peak amplitude, RMS energy, dynamic range).
- View the 4-grid visualizer:
  1. Time-Domain Waveform $y(t)$
  2. FFT Frequency Spectrum $|X(f)|$ in dB
  3. 20-band MFCC Spectrogram
  4. 128-band Log-Mel Spectrogram ($128 \times 216$)
- Use the **Play**, **Pause**, and **Stop** audio player controls.

#### 3. AI Prediction Tab
- Click **"Classify Loaded Audio"**.
- View Top-1 predicted class, confidence percentage, and inference latency.
- Inspect the Top-5 ranked classification table with visual confidence progress bars.

#### 4. Visual Translation Tab
- Select your desired generative rendering mode from the dropdown:
  - *Class-Specific Generative Geometry*
  - *Sound Mandala / Energy Aura*
  - *Frequency Harmonic Spiral*
  - *Waveform Geometric Flower*
  - *Acoustic Particle Constellation*
- Click **"Render Visual"** to compute the mathematical mapping and generate the artwork.
- Inspect the **Sound-to-Visual Telemetry** panel showing real-time physical feature mappings.
- Click **"Save Visual as PNG"** to export the artwork to `outputs/generated_visuals/`.

#### 5. Model Performance Tab
- View actual verified test metrics (Accuracy, Precision, Recall, F1-Score).
- Inspect high-resolution Training Loss/Accuracy curves and the 50-class Confusion Matrix.
- Retrain the model on-demand with live progress tracking.

