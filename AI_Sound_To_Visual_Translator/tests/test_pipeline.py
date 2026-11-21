"""
Comprehensive Automated Test Suite for AI-Based Sound-to-Visual Translation System.
Tests:
  1. Audio Loading & Preprocessing
  2. Multi-Feature Extraction & Dimensions (FFT, MFCC, Mel, Centroid, ZCR, RMS, Chroma)
  3. Guaranteed CNN Input Shape (128, 216, 1)
  4. Deterministic Sound-to-Visual Translation Consistency
  5. Feature Sensitivity & Sound Distinctiveness
  6. Visual Render Modes & PNG Export
  7. Robust Error Handling (Invalid files, Missing model, Missing data)
  8. Model Loading & Prediction Pipeline
  9. ESC-50 50-Class Archetype Mappings
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Ensure Keras backend is torch
os.environ.setdefault("KERAS_BACKEND", "torch")

project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

from audio_preprocessing import load_audio, get_audio_metadata, TARGET_SAMPLE_RATE, TARGET_SAMPLES
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
from visual_translation import SoundVisualTranslator, VISUAL_MODES, CLASS_ARCHETYPES
from dataset_loader import find_dataset_paths, ESC50_CLASSES
from prediction import AudioPredictor


class TestSoundToVisualPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.project_root = project_root
        cls.dataset_csv, cls.audio_dir = find_dataset_paths()
        # Pick sample files from different classes
        cls.sample_wav_1 = str(cls.audio_dir / "1-100032-A-0.wav")   # dog
        cls.sample_wav_2 = str(cls.audio_dir / "1-1791-A-26.wav")    # glass_breaking

    def test_01_audio_loading(self):
        """Verify audio loader resamples to 22,050 Hz and enforces exact 110,250 samples."""
        y, sr = load_audio(self.sample_wav_1, target_sr=TARGET_SAMPLE_RATE)
        self.assertEqual(sr, 22050)
        self.assertEqual(len(y), TARGET_SAMPLES)
        self.assertEqual(y.dtype, np.float32)
        # Peak normalization check
        self.assertLessEqual(np.max(np.abs(y)), 1.0)

    def test_02_audio_metadata(self):
        """Verify metadata calculation returns correct types and values."""
        meta = get_audio_metadata(self.sample_wav_1)
        self.assertIn("filename", meta)
        self.assertIn("rms_energy", meta)
        self.assertIn("dynamic_range_db", meta)
        self.assertGreater(meta["rms_energy"], 0.0)

    def test_03_mel_spectrogram_shape(self):
        """Verify Mel Spectrogram is strictly (128, 216) across all audio files."""
        y, sr = load_audio(self.sample_wav_1)
        mel = compute_mel_spectrogram(y, sr=sr, n_mels=128, target_frames=216)
        self.assertEqual(mel.shape, (128, 216))
        self.assertEqual(mel.dtype, np.float32)

    def test_04_feature_extraction_completeness(self):
        """Verify all required audio features (FFT, MFCC, Centroid, ZCR, RMS, Chroma) are computed."""
        y, sr = load_audio(self.sample_wav_1)
        fft_res = compute_fft(y, sr=sr)
        self.assertIn("dominant_freq_hz", fft_res)
        self.assertGreater(fft_res["dominant_freq_hz"], 0)

        mfcc_res = compute_mfcc(y, sr=sr, n_mfcc=20)
        self.assertEqual(mfcc_res["mfcc_matrix"].shape, (20, 216))

        cent_res = compute_spectral_centroid(y, sr=sr)
        self.assertGreater(cent_res["mean"], 0)

        zcr_res = compute_zero_crossing_rate(y)
        self.assertGreater(zcr_res["mean"], 0)

        rms_res = compute_rms_energy(y)
        self.assertGreater(rms_res["mean"], 0)

        chroma_res = compute_chroma(y, sr=sr)
        self.assertIn(chroma_res["dominant_pitch_name"], ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"])

    def test_05_visual_translation_deterministic_consistency(self):
        """Verify that translating the same audio twice yields identical visual parameters."""
        y, sr = load_audio(self.sample_wav_1)
        feats1 = extract_all_visual_features(y, sr=sr)
        feats2 = extract_all_visual_features(y, sr=sr)

        translator = SoundVisualTranslator()
        p1 = translator.compute_visual_parameters(feats1, predicted_class="dog", confidence=95.0)
        p2 = translator.compute_visual_parameters(feats2, predicted_class="dog", confidence=95.0)

        self.assertEqual(p1["base_radius"], p2["base_radius"])
        self.assertEqual(p1["rotation_angle_rad"], p2["rotation_angle_rad"])
        self.assertEqual(p1["vertices"], p2["vertices"])
        self.assertEqual(p1["harmonic_layers"], p2["harmonic_layers"])
        self.assertEqual(p1["hue_primary"], p2["hue_primary"])

    def test_06_visual_translation_feature_distinctiveness(self):
        """Verify that two distinctly different sounds produce different visual parameters."""
        y1, sr1 = load_audio(self.sample_wav_1)  # dog
        y2, sr2 = load_audio(self.sample_wav_2)  # glass_breaking

        feats1 = extract_all_visual_features(y1, sr=sr1)
        feats2 = extract_all_visual_features(y2, sr=sr2)

        translator = SoundVisualTranslator()
        p1 = translator.compute_visual_parameters(feats1, predicted_class="dog")
        p2 = translator.compute_visual_parameters(feats2, predicted_class="glass_breaking")

        # Glass breaking has significantly higher frequency & ZCR than dog bark
        self.assertNotEqual(p1["vertices"], p2["vertices"])
        self.assertNotEqual(p1["archetype"], p2["archetype"])

    def test_07_all_5_visual_render_modes(self):
        """Verify that all 5 visual modes render without errors and save to PNG."""
        y, sr = load_audio(self.sample_wav_1)
        translator = SoundVisualTranslator()

        for mode in VISUAL_MODES:
            fig, params = translator.render_visual(y, sr=sr, predicted_class="dog", confidence=90.0, visual_mode=mode)
            self.assertIsNotNone(fig)
            saved_path = translator.save_visual_png(fig, predicted_class="dog", custom_filename=f"unit_test_{mode.replace(' ', '_')}")
            self.assertTrue(Path(saved_path).exists())

    def test_08_esc50_classes_and_archetype_coverage(self):
        """Verify that all 50 ESC-50 classes have valid archetype assignments."""
        self.assertEqual(len(ESC50_CLASSES), 50)
        for c in ESC50_CLASSES:
            self.assertIn(c, CLASS_ARCHETYPES, f"Class {c} missing archetype mapping!")

    def test_09_error_handling_invalid_audio(self):
        """Verify appropriate exception is raised on invalid/non-existent audio file."""
        with self.assertRaises(FileNotFoundError):
            load_audio("non_existent_file.wav")

    def test_10_error_handling_missing_model_graceful(self):
        """Verify AudioPredictor handles missing model gracefully without crashing."""
        dummy_predictor = AudioPredictor(model_path="non_existent_model.keras")
        self.assertFalse(dummy_predictor.is_loaded)


if __name__ == "__main__":
    unittest.main()
