"""
AI-Based Sound-to-Visual Translation for Intelligent Audio Pattern Recognition
Source package initialization and public API exports.
"""

import os
import sys
from pathlib import Path

# Set global Keras backend to PyTorch before importing Keras anywhere
os.environ.setdefault("KERAS_BACKEND", "torch")

# Ensure src directory is in sys.path
_src_dir = str(Path(__file__).resolve().parent)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from audio_preprocessing import (
    load_audio,
    normalize_audio,
    get_audio_metadata,
    AudioPlayer,
    TARGET_SAMPLE_RATE
)
from feature_extraction import (
    compute_fft,
    compute_mel_spectrogram,
    compute_mfcc,
    compute_spectral_centroid,
    compute_zero_crossing_rate,
    compute_rms_energy,
    compute_chroma,
    extract_all_visual_features
)
from cnn_model import build_audio_cnn, set_seed, RANDOM_SEED
from dataset_loader import (
    load_or_create_dataset_cache,
    load_label_mapping,
    save_label_mapping,
    ESC50_CLASSES
)
from prediction import AudioPredictor, ESC10_CLASSES
from visual_translation import SoundVisualTranslator, VISUAL_MODES, CLASS_ARCHETYPES
from evaluation import (
    evaluate_predictions,
    plot_and_save_confusion_matrix,
    plot_and_save_training_history,
    save_metrics_summary
)
from hardware_info import get_system_hardware_info, get_hardware_summary_text

__version__ = "1.0.0"
__author__ = "Advanced AI Audio-Visual Engineering Team"

__all__ = [
    "load_audio",
    "normalize_audio",
    "get_audio_metadata",
    "AudioPlayer",
    "TARGET_SAMPLE_RATE",
    "compute_fft",
    "compute_mel_spectrogram",
    "compute_mfcc",
    "compute_spectral_centroid",
    "compute_zero_crossing_rate",
    "compute_rms_energy",
    "compute_chroma",
    "extract_all_visual_features",
    "build_audio_cnn",
    "set_seed",
    "RANDOM_SEED",
    "load_or_create_dataset_cache",
    "load_label_mapping",
    "save_label_mapping",
    "ESC50_CLASSES",
    "AudioPredictor",
    "ESC10_CLASSES",
    "SoundVisualTranslator",
    "VISUAL_MODES",
    "CLASS_ARCHETYPES",
    "evaluate_predictions",
    "plot_and_save_confusion_matrix",
    "plot_and_save_training_history",
    "save_metrics_summary",
    "get_system_hardware_info",
    "get_hardware_summary_text"
]
