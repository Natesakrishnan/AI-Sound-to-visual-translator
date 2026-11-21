# Installation Guide

## AI-Based Sound-to-Visual Translation for Intelligent Audio Pattern Recognition

This document provides complete, step-by-step instructions to set up the environment and install all dependencies on Windows.

---

### 1. Prerequisites

- **Operating System**: Windows 10 / 11 (64-bit)
- **Python Version**: Python 3.10 to 3.13 (64-bit)
- **System Memory**: 8 GB RAM minimum (16 GB recommended)
- **Hardware Acceleration**: NVIDIA GPU with CUDA support (e.g., RTX 3050 6GB) or modern multi-core CPU (Intel Core i5-13420H / AMD Ryzen)

---

### 2. Verify Python Installation

Open PowerShell or Command Prompt and verify that Python and `pip` are accessible:

```bash
python --version
pip --version
```

If Python is not recognized, ensure that Python is added to your Windows system `PATH`.

---

### 3. Install Required Dependencies

Navigate to the project root directory:

```bash
cd "K:\AI project\AI_Sound_To_Visual_Translator"
```

Install the dependencies via `pip`:

```bash
pip install -r requirements.txt
```

#### Primary Packages Installed:
- **`torch`**: Deep Learning runtime & CUDA engine
- **`keras`**: Modern Keras 3 high-level API
- **`librosa`**: Comprehensive audio analysis and Mel spectrogram generation
- **`soundfile`**: Fast lossless WAV audio I/O
- **`pygame`**: Desktop audio playback engine
- **`scipy` & `numpy`**: Fast Fourier Transform (FFT) and matrix algebra
- **`scikit-learn`**: Evaluation metrics and Confusion Matrix computation
- **`matplotlib` & `pillow`**: High-resolution embedded plotting and PNG art export
- **`pandas`**: ESC-50 dataset metadata parsing

---

### 4. Hardware & GPU Setup (NVIDIA RTX 3050 6GB)

The application automatically checks for CUDA GPU acceleration:
- If a compatible CUDA-enabled PyTorch build is present, training and inference execute on the **NVIDIA GeForce RTX 3050 Laptop GPU**.
- If CUDA is not configured, the system seamlessly falls back to the **13th Gen Intel Core i5-13420H CPU** without crashing or requiring manual code modification.

To check active hardware status:
```bash
python src/hardware_info.py
```

---

### 5. Verify Package Installations

Run the following quick verification script:

```bash
python -c "import torch, keras, librosa, soundfile, pygame, matplotlib, sklearn; print('All dependencies verified successfully!')"
```
