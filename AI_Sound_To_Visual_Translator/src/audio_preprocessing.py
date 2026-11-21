"""
Module 1: Audio Input & Preprocessing
Handles audio loading, mono conversion, resampling, duration standardization,
normalization, metadata calculation, and real-time audio playback controls.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import soundfile as sf
import librosa
import pygame


# Standardized Target Configuration
TARGET_SAMPLE_RATE = 22050
TARGET_DURATION = 5.0
TARGET_SAMPLES = int(TARGET_SAMPLE_RATE * TARGET_DURATION)  # 110,250 samples


def load_audio(
    file_path: str,
    target_sr: int = TARGET_SAMPLE_RATE,
    duration: float = TARGET_DURATION,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file, convert to mono, resample to target_sr, fix duration to exact target_samples,
    and optionally normalize amplitudes to [-1.0, 1.0].

    Args:
        file_path: Path to the audio file (.wav)
        target_sr: Desired sampling rate (default: 22050 Hz)
        duration: Desired duration in seconds (default: 5.0 s)
        mono: Whether to convert multi-channel audio to mono (default: True)
        normalize: Whether to apply peak normalization (default: True)

    Returns:
        Tuple of (y: np.ndarray [1D float32], sr: int)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # Read audio data using soundfile with librosa fallback
    try:
        y, orig_sr = sf.read(str(path), dtype="float32")
    except Exception:
        y, orig_sr = librosa.load(str(path), sr=None, mono=False)

    # Convert multi-channel to mono
    if y.ndim > 1:
        if mono:
            y = np.mean(y, axis=-1 if y.shape[-1] < y.shape[0] else 0)
        else:
            y = y[:, 0]  # First channel fallback

    # Resample if needed
    if orig_sr != target_sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
        sr = target_sr
    else:
        sr = orig_sr

    # Ensure target length (110,250 samples for 5.0s @ 22050Hz)
    expected_samples = int(target_sr * duration)
    y = fix_audio_length(y, expected_samples)

    # Normalization
    if normalize:
        y = normalize_audio(y, method="peak")

    return y.astype(np.float32), sr


def fix_audio_length(y: np.ndarray, target_length: int = TARGET_SAMPLES) -> np.ndarray:
    """
    Pad or crop audio array to an exact sample length using symmetric padding or center cropping.
    """
    current_length = len(y)
    if current_length == target_length:
        return y
    elif current_length < target_length:
        # Zero-pad symmetrically
        pad_total = target_length - current_length
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(y, (pad_left, pad_right), mode="constant", constant_values=0.0)
    else:
        # Center crop
        crop_start = (current_length - target_length) // 2
        return y[crop_start: crop_start + target_length]


def normalize_audio(y: np.ndarray, method: str = "peak", target_peak: float = 0.95) -> np.ndarray:
    """
    Normalize audio signal.
    Methods:
        'peak': Scale peak absolute amplitude to target_peak (e.g. 0.95).
        'rms': Scale RMS energy to a reference level.
    """
    if len(y) == 0:
        return y

    if method == "peak":
        max_val = np.max(np.abs(y))
        if max_val > 1e-7:
            return y * (target_peak / max_val)
        return y
    elif method == "rms":
        rms = np.sqrt(np.mean(y**2) + 1e-9)
        target_rms = 0.1
        return y * (target_rms / (rms + 1e-7))
    return y


def get_audio_metadata(file_path: str, y: Optional[np.ndarray] = None, sr: Optional[int] = None) -> Dict[str, Any]:
    """
    Extract comprehensive metadata and physical statistics from an audio file and its array.
    """
    path = Path(file_path)
    file_size_kb = path.stat().st_size / 1024.0 if path.exists() else 0.0

    raw_sr = 44100
    channels = 1
    raw_duration = 5.0
    subtype = "PCM_16"

    if path.exists():
        try:
            info = sf.info(str(path))
            raw_sr = info.samplerate
            channels = info.channels
            raw_duration = info.duration
            subtype = info.subtype
        except Exception:
            pass

    if y is None or sr is None:
        y, sr = load_audio(str(path), target_sr=TARGET_SAMPLE_RATE)

    # Compute signal metrics
    max_amp = float(np.max(y))
    min_amp = float(np.min(y))
    peak_abs = float(np.max(np.abs(y)))
    rms_energy = float(np.sqrt(np.mean(y**2)))
    dynamic_range_db = float(20 * np.log10(peak_abs / (rms_energy + 1e-8) + 1e-6))
    zero_crossings = int(np.sum(np.abs(np.diff(np.sign(y))) > 0) // 2)

    return {
        "filename": path.name,
        "filepath": str(path.resolve()),
        "filesize_kb": round(file_size_kb, 1),
        "raw_sample_rate": raw_sr,
        "processed_sample_rate": sr,
        "channels": channels,
        "raw_duration_sec": round(raw_duration, 2),
        "processed_duration_sec": round(len(y) / sr, 2),
        "total_samples": len(y),
        "max_amplitude": round(max_amp, 4),
        "min_amplitude": round(min_amp, 4),
        "peak_amplitude": round(peak_abs, 4),
        "rms_energy": round(rms_energy, 4),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "zero_crossings": zero_crossings,
        "subtype": subtype
    }


class AudioPlayer:
    """
    Audio Player helper using pygame.mixer for playback in desktop GUI applications.
    """
    _initialized = False

    def __init__(self):
        self.current_file = None
        self.is_playing = False
        self.is_paused = False
        self.init_mixer()

    @classmethod
    def init_mixer(cls):
        if not cls._initialized:
            try:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
                cls._initialized = True
            except Exception as e:
                print(f"Pygame mixer init warning: {e}")

    def load(self, file_path: str) -> bool:
        """Load audio file into the player."""
        if not Path(file_path).exists():
            return False
        try:
            pygame.mixer.music.load(file_path)
            self.current_file = file_path
            self.is_playing = False
            self.is_paused = False
            return True
        except Exception as e:
            print(f"AudioPlayer load error: {e}")
            return False

    def play(self) -> bool:
        """Start or resume audio playback."""
        if not self.current_file:
            return False
        try:
            if self.is_paused:
                pygame.mixer.music.unpause()
                self.is_paused = False
                self.is_playing = True
            else:
                pygame.mixer.music.play()
                self.is_playing = True
                self.is_paused = False
            return True
        except Exception as e:
            print(f"AudioPlayer play error: {e}")
            return False

    def pause(self):
        """Pause playback."""
        if self.is_playing and not self.is_paused:
            pygame.mixer.music.pause()
            self.is_paused = True
            self.is_playing = False

    def stop(self):
        """Stop playback completely."""
        try:
            pygame.mixer.music.stop()
            self.is_playing = False
            self.is_paused = False
        except Exception:
            pass

    def get_busy(self) -> bool:
        """Check if audio is currently playing."""
        try:
            return pygame.mixer.music.get_busy()
        except Exception:
            return False


if __name__ == "__main__":
    sample_wav = r"K:/AI project/Datasets/audio/1-100032-A-0.wav"
    if os.path.exists(sample_wav):
        y_data, s_rate = load_audio(sample_wav)
        meta = get_audio_metadata(sample_wav, y_data, s_rate)
        print("Preprocessed Audio Shape:", y_data.shape)
        print("Metadata:", meta)
