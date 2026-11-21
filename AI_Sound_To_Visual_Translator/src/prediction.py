"""
Module 3: Audio Prediction Pipeline
Performs CNN-based inference for audio classification on single WAV files.
Returns predicted class, confidence scores, and Top-5 class rankings.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

from audio_preprocessing import load_audio, TARGET_SAMPLE_RATE
from feature_extraction import compute_mel_spectrogram
from dataset_loader import load_label_mapping, ESC50_CLASSES


# Official ESC-10 subset classes for badge tagging
ESC10_CLASSES = {
    "dog", "rooster", "rain", "sea_waves", "crying_baby",
    "sneezing", "clock_tick", "helicopter", "chainsaw", "crackling_fire"
}


class AudioPredictor:
    """
    Inference Engine for ESC-50 Environmental Audio Classification.
    Loads and runs the trained Experiment 4 CNN model with fallback support.
    """
    def __init__(self, model_path: Optional[str] = None, label_map_path: Optional[str] = None):
        self.project_root = Path(__file__).resolve().parent.parent

        # Prefer trained Experiment 4 validation-selected checkpoint (71.00% accuracy)
        exp4_model = self.project_root / "experiments" / "task509_reproduction_experiment4" / "models" / "audio_cnn_model_task509_best_val_loss.keras"
        default_model = self.project_root / "models" / "audio_cnn_model.keras"

        if model_path:
            self.model_path = Path(model_path)
        elif exp4_model.exists():
            self.model_path = exp4_model
        else:
            self.model_path = default_model

        exp4_label_map = self.project_root / "experiments" / "task509_reproduction_experiment4" / "models" / "label_mapping.json"
        default_label_map = self.project_root / "models" / "label_mapping.json"

        if label_map_path:
            self.label_map_path = Path(label_map_path)
        elif exp4_label_map.exists():
            self.label_map_path = exp4_label_map
        else:
            self.label_map_path = default_label_map

        self.model: Optional[keras.Model] = None
        self.id_to_class: Dict[int, str] = {}
        self.class_to_id: Dict[str, int] = {}
        self.is_loaded: bool = False

        self.load_resources()

    def load_resources(self, force_reload: bool = False) -> bool:
        """Attempt to load trained Keras model and label mapping once into memory."""
        if self.is_loaded and not force_reload:
            return True

        if not self.model_path.exists() or not self.label_map_path.exists():
            self.is_loaded = False
            return False

        try:
            self.model = keras.models.load_model(str(self.model_path))
            self.id_to_class = load_label_mapping(str(self.label_map_path))
            self.class_to_id = {v: k for k, v in self.id_to_class.items()}
            self.is_loaded = True
            print(f"[AudioPredictor] Loaded model from: {self.model_path}")
            print(f"[AudioPredictor] Loaded {len(self.id_to_class)} classes from: {self.label_map_path}")
            return True
        except Exception as e:
            print(f"[AudioPredictor] Error loading model or labels: {e}")
            self.is_loaded = False
            return False

    def predict(self, audio_file_path: str) -> Dict[str, Any]:
        """
        Execute full prediction pipeline on a single audio file.

        Pipeline:
            Audio File (.wav) -> Load & Preprocess -> Mel Spectrogram (1, 128, 216, 1)
            -> CNN Forward Pass -> Softmax Probabilities -> Top-5 Predictions.

        Returns:
            Dictionary with predicted class, confidence, top 5 classes, and inference latency.
        """
        if not self.is_loaded:
            # Try reloading once
            if not self.load_resources():
                raise RuntimeError(
                    f"Model is not loaded! Model file not found at: {self.model_path}\n"
                    "Please train the model first using the 'Train Model' tab or train_model.py."
                )

        t_start = time.time()

        # 1. Load and standardize audio
        y, sr = load_audio(audio_file_path, target_sr=TARGET_SAMPLE_RATE)

        # 2. Extract standardized Mel Spectrogram using Experiment 4 decibel normalization
        mel = compute_mel_spectrogram(y, sr=sr, n_mels=128, target_frames=216, norm_method="decibel_clip")
        mel_input = np.expand_dims(np.expand_dims(mel, axis=0), axis=-1).astype(np.float32)

        # 3. Model Inference (Fast direct forward pass on GPU/CPU)
        raw_output = self.model(mel_input, training=False)
        if hasattr(raw_output, "cpu"):
            probabilities = raw_output.cpu().detach().numpy()[0]
        elif hasattr(raw_output, "numpy"):
            probabilities = raw_output.numpy()[0]
        else:
            probabilities = np.array(raw_output)[0]
        inference_time_ms = round((time.time() - t_start) * 1000.0, 2)

        # 4. Extract Top-1 and Top-5 predictions
        top_indices = np.argsort(probabilities)[::-1][:5]
        top_1_idx = int(top_indices[0])
        top_1_class = self.id_to_class.get(top_1_idx, ESC50_CLASSES[top_1_idx] if top_1_idx < len(ESC50_CLASSES) else "unknown")
        top_1_confidence = float(probabilities[top_1_idx])

        top_5_list: List[Tuple[str, float, bool]] = []
        for idx in top_indices:
            c_name = self.id_to_class.get(int(idx), f"class_{idx}")
            conf = float(probabilities[idx])
            is_esc10 = c_name in ESC10_CLASSES
            top_5_list.append((c_name, round(conf * 100.0, 2), is_esc10))

        return {
            "predicted_class": top_1_class,
            "confidence_pct": round(top_1_confidence * 100.0, 2),
            "confidence_raw": top_1_confidence,
            "top_5": top_5_list,
            "is_esc10": top_1_class in ESC10_CLASSES,
            "inference_time_ms": inference_time_ms,
            "probabilities": probabilities,
            "raw_audio_shape": y.shape,
            "mel_spectrogram": mel
        }


if __name__ == "__main__":
    predictor = AudioPredictor()
    print("Predictor loaded:", predictor.is_loaded)
    sample_wav = r"K:/AI project/Datasets/audio/1-100032-A-0.wav"
    if os.path.exists(sample_wav) and predictor.is_loaded:
        res = predictor.predict(sample_wav)
        print("Prediction Result:", res["predicted_class"], res["confidence_pct"], "%")
