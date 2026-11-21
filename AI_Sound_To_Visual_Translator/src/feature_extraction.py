"""
Module 2: Audio Feature Extraction
Extracts FFT, Mel Spectrogram, MFCC, Spectral Centroid, Zero Crossing Rate,
RMS Energy, Chroma, and visual parameters.
"""

from typing import Dict, Any, Tuple
import numpy as np
import scipy.fft
import librosa


def compute_fft(y: np.ndarray, sr: int = 22050) -> Dict[str, Any]:
    """
    Compute Fast Fourier Transform (FFT) on the audio signal to determine its frequency spectrum.

    Returns:
        Dictionary with positive frequencies, magnitude spectrum (dB),
        dominant frequency (Hz), spectral rolloff, and spectral flatness.
    """
    n = len(y)
    # Apply Hanning window to reduce spectral leakage
    windowed = y * np.hanning(n)
    fft_vals = scipy.fft.rfft(windowed)
    freqs = scipy.fft.rfftfreq(n, d=1.0 / sr)
    magnitude = np.abs(fft_vals)

    # Convert to dB relative to peak
    max_mag = np.max(magnitude) + 1e-9
    magnitude_db = 20 * np.log10(magnitude / max_mag + 1e-6)

    # Find dominant peak frequency
    peak_idx = np.argmax(magnitude[1:]) + 1  # Skip 0 Hz DC offset
    dominant_freq = float(freqs[peak_idx])

    # Spectral centroid from FFT
    spectral_centroid_fft = float(np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-9))

    # Downsample FFT spectrum for lightweight plotting (e.g. 512 bins)
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
    target_frames: int = 216,
    norm_method: str = "decibel_clip"
) -> np.ndarray:
    """
    Generate Log-Mel Spectrogram matrix with guaranteed exact shape (n_mels, target_frames),
    i.e. (128, 216).

    Default (norm_method="decibel_clip"):
        Applies exact Task 509 / Experiment 4 decibel standardization into [-1.0, +1.0]:
        np.clip((mel_db + 40.0) / 40.0, -1.0, 1.0)
        Matches the input distribution of the trained Experiment 4 CNN model.

    Optional (norm_method="zscore"):
        Per-sample instance standardization (zero mean, unit variance).
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

    if norm_method == "decibel_clip":
        # Standardize dB scale [-80.0, 0.0] into normalized [-1.0, +1.0] distribution
        # Peak (0 dB) -> +1.0, Middle (-40 dB) -> 0.0, Noise floor (-80 dB) -> -1.0
        mel_norm = np.clip((mel_db + 40.0) / 40.0, -1.0, 1.0)
    elif norm_method == "zscore":
        mel_mean = np.mean(mel_db)
        mel_std = np.std(mel_db) + 1e-6
        mel_norm = (mel_db - mel_mean) / mel_std
    else:
        mel_norm = mel_db

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


def compute_mfcc(
    y: np.ndarray,
    sr: int = 22050,
    n_mfcc: int = 20,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    target_frames: int = 216
) -> Dict[str, Any]:
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCCs).

    Returns:
        Dictionary containing MFCC matrix of shape (n_mfcc, target_frames),
        mean coefficients, and std coefficients.
    """
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    if mfcc.shape[1] < target_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, target_frames - mfcc.shape[1])), mode="edge")
    else:
        mfcc = mfcc[:, :target_frames]

    return {
        "mfcc_matrix": mfcc.astype(np.float32),
        "mean_mfcc": np.mean(mfcc, axis=1),
        "std_mfcc": np.std(mfcc, axis=1)
    }


def compute_spectral_centroid(
    y: np.ndarray,
    sr: int = 22050,
    n_fft: int = 2048,
    hop_length: int = 512
) -> Dict[str, Any]:
    """Calculate time-varying spectral centroid and its summary statistics."""
    centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length
    )[0]
    return {
        "series": centroid,
        "mean": float(np.mean(centroid)),
        "std": float(np.std(centroid)),
        "min": float(np.min(centroid)),
        "max": float(np.max(centroid))
    }


def compute_zero_crossing_rate(y: np.ndarray, hop_length: int = 512) -> Dict[str, Any]:
    """Calculate time-varying Zero Crossing Rate (ZCR)."""
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    return {
        "series": zcr,
        "mean": float(np.mean(zcr)),
        "std": float(np.std(zcr)),
        "max": float(np.max(zcr))
    }


def compute_rms_energy(y: np.ndarray, hop_length: int = 512) -> Dict[str, Any]:
    """Calculate time-varying RMS energy envelope."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    return {
        "series": rms,
        "mean": float(np.mean(rms)),
        "std": float(np.std(rms)),
        "max": float(np.max(rms))
    }


def compute_chroma(
    y: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    target_frames: int = 216
) -> Dict[str, Any]:
    """
    Calculate 12-element Chroma STFT features (pitch classes C, C#, D, ..., B).
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
    if chroma.shape[1] < target_frames:
        chroma = np.pad(chroma, ((0, 0), (0, target_frames - chroma.shape[1])), mode="edge")
    else:
        chroma = chroma[:, :target_frames]

    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    chroma_mean = np.mean(chroma, axis=1)
    dominant_pitch_idx = int(np.argmax(chroma_mean))
    dominant_pitch_name = pitch_names[dominant_pitch_idx]

    return {
        "chroma_matrix": chroma.astype(np.float32),
        "chroma_mean": chroma_mean,
        "dominant_pitch_idx": dominant_pitch_idx,
        "dominant_pitch_name": dominant_pitch_name
    }


def extract_all_visual_features(y: np.ndarray, sr: int = 22050) -> Dict[str, Any]:
    """
    Extract and normalize all acoustic parameters into a structured dictionary
    specifically designed for the deterministic Sound-to-Visual Translation Engine.
    """
    fft_info = compute_fft(y, sr=sr)
    rms_info = compute_rms_energy(y)
    zcr_info = compute_zero_crossing_rate(y)
    centroid_info = compute_spectral_centroid(y, sr=sr)
    chroma_info = compute_chroma(y, sr=sr)

    # Normalize parameters into [0.0, 1.0] for stable geometric translation
    # 1. RMS Energy Normalization (Typical audio RMS ranges from 0.01 to 0.40)
    rms_norm = float(np.clip(rms_info["mean"] / 0.35, 0.05, 1.0))
    rms_max_norm = float(np.clip(rms_info["max"] / 0.60, 0.05, 1.0))

    # 2. Spectral Centroid Normalization (Nyquist = sr/2 = 11025 Hz, Typical centroid = 500 - 6000 Hz)
    centroid_norm = float(np.clip(centroid_info["mean"] / 6000.0, 0.02, 1.0))

    # 3. ZCR Normalization (Typical ZCR ranges from 0.01 to 0.30)
    zcr_norm = float(np.clip(zcr_info["mean"] / 0.25, 0.02, 1.0))

    # 4. Dominant Frequency Normalization (0 to 5000 Hz)
    dom_freq_norm = float(np.clip(fft_info["dominant_freq_hz"] / 4000.0, 0.01, 1.0))

    # 5. Dynamic Contrast / Peak Energy
    peak_amp = float(np.max(np.abs(y)))

    return {
        # Raw physical values
        "raw_rms_mean": round(rms_info["mean"], 4),
        "raw_rms_max": round(rms_info["max"], 4),
        "raw_zcr_mean": round(zcr_info["mean"], 4),
        "raw_centroid_hz": round(centroid_info["mean"], 1),
        "raw_dominant_freq_hz": round(fft_info["dominant_freq_hz"], 1),
        "dominant_pitch": chroma_info["dominant_pitch_name"],
        "dominant_pitch_idx": chroma_info["dominant_pitch_idx"],
        "peak_amplitude": round(peak_amp, 4),

        # Normalized values for geometry mapping [0.0, 1.0]
        "rms_norm": rms_norm,
        "rms_max_norm": rms_max_norm,
        "centroid_norm": centroid_norm,
        "zcr_norm": zcr_norm,
        "dom_freq_norm": dom_freq_norm,
        "chroma_mean": chroma_info["chroma_mean"],

        # Time series for temporal visual modulation
        "rms_series": rms_info["series"],
        "zcr_series": zcr_info["series"],
        "centroid_series": centroid_info["series"],
        "chroma_matrix": chroma_info["chroma_matrix"]
    }


if __name__ == "__main__":
    from audio_preprocessing import load_audio
    sample_wav = r"K:/AI project/Datasets/audio/1-100032-A-0.wav"
    y_data, sr_val = load_audio(sample_wav)
    mel_mat = compute_mel_spectrogram(y_data, sr_val)
    print("Mel Spectrogram Shape:", mel_mat.shape)
    v_feats = extract_all_visual_features(y_data, sr_val)
    print("Visual Features Summary:", {k: v for k, v in v_feats.items() if not isinstance(v, np.ndarray)})
