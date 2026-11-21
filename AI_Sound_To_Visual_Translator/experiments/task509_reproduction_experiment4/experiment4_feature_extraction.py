"""
Module: Task 509 Feature Extraction (Experiment 4)
Exact implementation from Step 389 of Task 509.
Converts audio into Log-Mel Spectrogram normalized to [-1.0, +1.0] zero-centered dynamic range.
"""

from typing import Dict, Any
import numpy as np
import scipy.fft
import librosa


def compute_fft(y: np.ndarray, sr: int = 22050) -> Dict[str, Any]:
    """Compute Fast Fourier Transform (FFT) on audio signal."""
    n = len(y)
    windowed = y * np.hanning(n)
    fft_vals = scipy.fft.rfft(windowed)
    freqs = scipy.fft.rfftfreq(n, d=1.0 / sr)
    magnitude = np.abs(fft_vals)

    max_mag = np.max(magnitude) + 1e-9
    magnitude_db = 20 * np.log10(magnitude / max_mag + 1e-6)

    peak_idx = np.argmax(magnitude[1:]) + 1
    dominant_freq = float(freqs[peak_idx])
    spectral_centroid_fft = float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-9))

    plot_bins = 512
    if len(freqs) > plot_bins:
        indices = np.linspace(0, len(freqs) - 1, plot_bins, dtype=int)
        plot_freqs = freqs[indices]
        plot_mags = magnitude_db[indices]
    else:
        plot_freqs = freqs
        plot_mags = magnitude_db

    return {
        "freqs": freqs,
        "magnitude": magnitude,
        "magnitude_db": magnitude_db,
        "plot_freqs": plot_freqs,
        "plot_magnitude_db": plot_mags,
        "dominant_freq_hz": round(dominant_freq, 1),
        "spectral_centroid_hz": round(spectral_centroid_fft, 1)
    }


def compute_mel_spectrogram(
    y: np.ndarray,
    sr: int = 22050,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    target_frames: int = 216
) -> np.ndarray:
    """
    Generate Log-Mel Spectrogram matrix with guaranteed exact shape (n_mels, target_frames),
    i.e. (128, 216), normalized into a zero-centered [-1.0, +1.0] dynamic range.
    EXACT Task 509 implementation (Step 389).
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=20,
        fmax=sr // 2,
        power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Standardize dB scale [-80.0, 0.0] into normalized [-1.0, +1.0] distribution
    # Peak (0 dB) -> +1.0, Middle (-40 dB) -> 0.0, Noise floor (-80 dB) -> -1.0
    mel_norm = np.clip((mel_db + 40.0) / 40.0, -1.0, 1.0)

    # Ensure exact target frames (216)
    current_frames = mel_norm.shape[1]
    if current_frames == target_frames:
        return mel_norm.astype(np.float32)
    elif current_frames < target_frames:
        pad_width = target_frames - current_frames
        mel_norm = np.pad(mel_norm, ((0, 0), (0, pad_width)), mode="edge")
    else:
        mel_norm = mel_norm[:, :target_frames]

    return mel_norm.astype(np.float32)

