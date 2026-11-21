# Project Architecture & Viva-Friendly Technical Documentation

## AI-Based Sound-to-Visual Translation for Intelligent Audio Pattern Recognition

---

### 1. High-Level Architectural Flow

```text
                             Input Audio (.wav)
                                     ↓
                     [Module 1: Audio Preprocessing]
                     • Resample to 22,050 Hz Mono
                     • Length Standardization (110,250 samples)
                     • Peak Normalization
                                     ↓
                     [Module 2: Feature Extraction]
                     • Log-Mel Spectrogram (128 Mel bands x 216 frames)
                     • Fast Fourier Transform (FFT) & Dominant Frequency
                     • 20-band Mel-Frequency Cepstral Coefficients (MFCC)
                     • Spectral Centroid, RMS Energy, Zero Crossing Rate (ZCR)
                     • 12-dimensional Chroma Pitch STFT
                                     ↓
                     [Module 3: CNN Audio Pattern Recognition]
                     • 4 Convolutional Blocks (Conv2D + BatchNorm + MaxPool + Dropout)
                     • Global Average Pooling + Dense Classification Head
                     • Softmax Probabilities over 50 ESC-50 Environmental Sound Classes
                                     ↓
                     [Module 4: Sound-to-Visual Translation]
                     • Deterministic Physical Feature-to-Visual Geometry Mapping
                     • Mathematical parameterization of Radius, Rotation, Complexity & Color
                     • Archetypal Class Mapping for 50 ESC-50 Categories
                                     ↓
                     Generated High-Resolution Visual Artwork & PNG Export
```

---

### 2. Detailed Module Breakdown

#### Module 1: Audio Input & Preprocessing (`src/audio_preprocessing.py`)
- **Signal Normalization**: Audio amplitudes are normalized to peak scale $[-0.95, +0.95]$ to ensure numerical stability.
- **Duration Standardization**: ESC-50 files are 5.0 seconds long ($N = 110,250$ samples at $22,050\text{ Hz}$). If an uploaded file is longer, it is center-cropped; if shorter, it is symmetrically zero-padded.
- **Audio Playback Engine**: Multi-threaded non-blocking audio playback powered by `pygame.mixer`.

#### Module 2: Audio Feature Extraction (`src/feature_extraction.py`)
1. **FFT (Fast Fourier Transform)**: Converts time-domain signal $y(t)$ to discrete frequency spectrum $X(f)$ using $O(N \log N)$ Cooley-Tukey algorithm.
2. **Log-Mel Spectrogram**: Mimics the non-linear frequency perception of human hearing using triangular Mel filters:
   $$m = 2595 \log_{10}\left(1 + \frac{f}{700}\right)$$
   Standardized to shape $(128, 216)$, where $n\_\text{mels}=128$, $n\_\text{fft}=2048$, and $\text{hop\_length}=512$.
3. **MFCC**: Captures timbral envelope characteristics via Discrete Cosine Transform (DCT) of the Mel filterbank energies.
4. **Spectral Centroid**: The "center of mass" of the audio spectrum, indicating perceptual brightness:
   $$\text{Centroid} = \frac{\sum_{k} f_k |X(k)|}{\sum_k |X(k)|}$$
5. **Zero-Crossing Rate (ZCR)**: Frequency of sign changes in the signal, identifying noise vs tonal harmonic sounds.
6. **Chroma STFT**: Projects spectral energy into 12 semitone pitch classes (C, C#, D, ..., B) to detect musical harmony.

#### Module 3: AI-Based Audio Pattern Recognition (`src/cnn_model.py`, `src/train_model.py`, `src/prediction.py`)
- **CNN Architecture**:
  ```text
  Input Tensor: (128, 216, 1)
       ↓
  Conv2D (32 filters, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> SpatialDropout(0.15)
       ↓
  Conv2D (64 filters, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> SpatialDropout(0.20)
       ↓
  Conv2D (128 filters, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> SpatialDropout(0.25)
       ↓
  Conv2D (256 filters, 3x3) -> BatchNorm -> ReLU -> MaxPool2D(2x2) -> SpatialDropout(0.30)
       ↓
  GlobalAveragePooling2D
       ↓
  Dense (256, ReLU) -> Dropout(0.40)
       ↓
  Dense (128, ReLU) -> Dropout(0.30)
       ↓
  Dense (50, Softmax) -> 50 Class Probabilities
  ```
- **Fold-Based Dataset Partitioning**:
  - **Folds 1, 2, 3** $\rightarrow$ Training Set (1,200 samples)
  - **Fold 4** $\rightarrow$ Validation Set (400 samples)
  - **Fold 5** $\rightarrow$ Unseen Final Test Set (400 samples)
- **Feature Caching**: Precomputed features stored in `outputs/cache/esc50_features_cache.npz` to eliminate repeated extraction overhead.

#### Module 4: Sound-to-Visual Translation Engine (`src/visual_translation.py`)
This module implements deterministic mathematical transformations from physical acoustic parameters to generative geometry:

1. **RMS Energy $\rightarrow$ Visual Radius & Scale**:
   $$R_{\text{base}} = 1.0 + 1.4 \times \text{RMS}_{\text{norm}}$$
2. **Spectral Centroid $\rightarrow$ Angular Rotation**:
   $$\theta_{\text{rot}} = 2\pi \times \text{Centroid}_{\text{norm}}$$
3. **Zero Crossing Rate $\rightarrow$ Geometric Complexity / Vertices**:
   $$N_{\text{vertices}} = 4 + \lfloor 28 \times \text{ZCR}_{\text{norm}} \rfloor$$
4. **Dominant Frequency $\rightarrow$ Harmonic Layer Count**:
   $$N_{\text{layers}} = 2 + \lfloor 6 \times \text{Freq}_{\text{norm}} \rfloor$$
5. **Chroma Pitch Class $\rightarrow$ Chromatic Palette**:
   $$\text{Hue}_{\text{primary}} = \frac{\text{PitchIndex} \times 30^\circ}{360^\circ}$$
6. **Class-Specific Generative Geometry Archetypes**:
   - **Animal Sounds**: Organic concentric orbital auras.
   - **Nature / Water Sounds**: Flowing streamline fluid cascades.
   - **Human Sounds**: Vocal tract acoustic standing wave nodes.
   - **Interior / Domestic Sounds**: Rhythmic mechanical lattice grids.
   - **Urban / Mechanical Sounds**: High-energy crystalline polygonal bursts.

---

### 3. Key Viva Questions & Answers

**Q1: Why is this project different from a simple audio classifier?**
> *Answer*: Standard audio classifiers merely output a text label. This system extracts multi-domain physical audio features, performs deep neural classification, and translates acoustic physics (energy, centroid, pitch, noisiness) into deterministic, explainable mathematical visual patterns.

**Q2: Why use Log-Mel Spectrograms instead of raw waveforms for CNN input?**
> *Answer*: Raw waveforms have high temporal redundancy and phase sensitivity. Mel spectrograms compress the acoustic information into a 2D time-frequency representation that mirrors human cochlear perception, allowing standard 2D Convolutional kernels to detect localized spectral patterns across time.

**Q3: How is data leakage prevented during training and evaluation?**
> *Answer*: The ESC-50 dataset is split strictly by predefined recording folds. Folds 1–3 are used exclusively for training, Fold 4 is used for validation monitoring, and Fold 5 is held out as a strictly unseen test set for final performance reporting and confusion matrix generation.

---

### 4. Experiment 4 Reproduction & GUI Integration Results

#### Benchmark Performance Summary
- **Trained Model Checkpoint**: `experiments/task509_reproduction_experiment4/models/audio_cnn_model_task509_best_val_loss.keras` (Epoch 38)
- **Held-Out Test Set**: ESC-50 Fold 5 (400 samples, 8 samples per class across 50 classes)
- **Test Accuracy**: **71.00%** (284 / 400 correct predictions)
- **Test Macro Precision**: **70.62%**
- **Test Macro Recall**: **71.00%**
- **Test Macro F1-Score**: **69.24%**
- **Inference Latency**: **32.8 ms** (NVIDIA RTX 3050 GPU, steady-state)
- **Model Load Time**: **0.432 s**

#### Complete Desktop GUI Workflow
1. **Tab 1: Home**: Real-time system hardware telemetry (CPU, GPU, RAM, Keras PyTorch engine), model readiness badge, and quick action navigation.
2. **Tab 2: Audio Analysis**: 4-grid signal plots (Waveform, FFT Spectrum, MFCC Matrix, Log-Mel Spectrogram) and 8 physical acoustic metrics (Duration, Sample Rate, RMS, Peak Amp, Centroid, ZCR, Dominant Frequency, Dominant Pitch) with Pygame audio playback controls.
3. **Tab 3: AI Prediction**: Background threaded execution (`threading.Thread`) with indeterminate progress bar, Top-1 class Hero card, confidence %, ESC-10 badge, Top-5 probability bars summing to 1.0, and latency display.
4. **Tab 4: Visual Translation**: 5 geometric archetypes (`animal_orbital`, `fluid_wave`, `harmonic_node`, `lattice_grid`, `angular_burst`), 5 selectable generative modes, real-time physical telemetry parameters, and high-resolution PNG artwork export.
5. **Tab 5: Model Performance**: Experiment 4 evaluation metrics, side-by-side training history curves (`training_history.png`), and full 50-class normalized confusion matrix (`confusion_matrix.png`).
