"""
Automated Verification & Integration Test Suite for Experiment 4 Model GUI Integration.
Tests:
1. Model loading & resource validation (Experiment 4 checkpoint & 50-class label map)
2. Full end-to-end pipeline across 3 diverse audio categories (Animal, Natural, Urban)
3. SoundVisualTranslator rendering across multiple modes and PNG export
4. GUI Application structure, tab initialization, and programmatic execution
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np

# Ensure project root and src are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("KERAS_BACKEND", "torch")

from audio_preprocessing import load_audio, get_audio_metadata, TARGET_SAMPLE_RATE, TARGET_SAMPLES
from feature_extraction import compute_mel_spectrogram, compute_fft, compute_mfcc, extract_all_visual_features
from prediction import AudioPredictor, ESC10_CLASSES
from visual_translation import SoundVisualTranslator, VISUAL_MODES
from gui import SoundVisualizerApp


def test_model_loading():
    print("\n" + "=" * 70)
    print("STEP 1: Testing Model Loading & Checkpoint Verification")
    print("=" * 70)

    t0 = time.time()
    predictor = AudioPredictor()
    load_time = time.time() - t0

    print(f"Model Load Time: {load_time:.3f} s (Target: < 5.0 s)")
    assert load_time < 5.0, f"Model loading too slow: {load_time} s"
    assert predictor.is_loaded, "Predictor failed to load model!"
    
    ckpt_path = str(predictor.model_path)
    print(f"Loaded Checkpoint Path: {ckpt_path}")
    assert "audio_cnn_model_task509_best_val_loss.keras" in ckpt_path or "task509_reproduction_experiment4" in ckpt_path, \
        f"Incorrect checkpoint loaded: {ckpt_path}"
    
    print(f"Classes Loaded: {len(predictor.id_to_class)} classes")
    assert len(predictor.id_to_class) == 50, f"Expected 50 classes, got {len(predictor.id_to_class)}"
    print("✓ Model Checkpoint & Label Mapping verified successfully!")
    return predictor


def test_end_to_end_samples(predictor: AudioPredictor):
    print("\n" + "=" * 70)
    print("STEP 2: Testing End-to-End Workflow on 3 Diverse ESC-50 Samples")
    print("=" * 70)

    audio_base = Path("K:/AI project/Datasets/audio")
    samples = [
        {"name": "Sample 1: Animal (Dog Bark)", "file": "1-100032-A-0.wav", "expected_category": "dog"},
        {"name": "Sample 2: Natural Soundscape (Rain)", "file": "1-17367-A-10.wav", "expected_category": "rain"},
        {"name": "Sample 3: Urban / Interior (Clock Tick)", "file": "1-21934-A-38.wav", "expected_category": "clock_tick"}
    ]

    translator = SoundVisualTranslator(output_dir=str(PROJECT_ROOT / "outputs" / "generated_visuals"))
    results = []

    for s in samples:
        print(f"\n--- Testing {s['name']} [{s['file']}] ---")
        file_path = audio_base / s["file"]
        assert file_path.exists(), f"Audio file not found: {file_path}"

        # 1. Audio Loading & Preprocessing
        t_load = time.time()
        y, sr = load_audio(str(file_path), target_sr=TARGET_SAMPLE_RATE)
        meta = get_audio_metadata(str(file_path), y, sr)
        load_latency = (time.time() - t_load) * 1000.0

        print(f"  • Preprocessed Shape: {y.shape} (Expected: ({TARGET_SAMPLES},))")
        print(f"  • Sample Rate: {sr} Hz")
        print(f"  • Peak Amplitude: {meta['peak_amplitude']:.4f} (Target <= 0.95)")
        print(f"  • RMS Energy: {meta['rms_energy']:.4f}")
        assert y.shape == (TARGET_SAMPLES,), f"Invalid audio shape: {y.shape}"
        assert sr == TARGET_SAMPLE_RATE, f"Invalid sr: {sr}"
        assert meta["peak_amplitude"] <= 0.9501, f"Peak amplitude exceeded: {meta['peak_amplitude']}"

        # 2. Feature Extraction
        t_feat = time.time()
        mel = compute_mel_spectrogram(y, sr=sr, n_mels=128, target_frames=216, norm_method="decibel_clip")
        v_feats = extract_all_visual_features(y, sr)
        feat_latency = (time.time() - t_feat) * 1000.0

        print(f"  • Mel Spectrogram Shape: {mel.shape} (Expected: (128, 216))")
        print(f"  • Mel Value Range: [{mel.min():.2f}, {mel.max():.2f}] (Expected [-1.0, 1.0])")
        print(f"  • Spectral Centroid: {v_feats['raw_centroid_hz']:.1f} Hz")
        print(f"  • Zero Crossing Rate: {v_feats['raw_zcr_mean']:.4f}")
        print(f"  • Dominant Freq: {v_feats['raw_dominant_freq_hz']:.1f} Hz")
        print(f"  • Dominant Pitch: {v_feats['dominant_pitch']}")
        assert mel.shape == (128, 216), f"Invalid mel shape: {mel.shape}"
        assert mel.min() >= -1.01 and mel.max() <= 1.01, f"Mel values out of range: [{mel.min()}, {mel.max()}]"

        # 3. CNN Prediction
        t_pred = time.time()
        pred = predictor.predict(str(file_path))
        pred_latency = (time.time() - t_pred) * 1000.0

        prob_sum = float(np.sum(pred["probabilities"]))
        print(f"  • Top-1 Predicted Class: '{pred['predicted_class']}' ({pred['confidence_pct']:.2f}%)")
        print(f"  • Inference Latency: {pred['inference_time_ms']:.2f} ms (Measured: {pred_latency:.2f} ms)")
        print(f"  • Probability Sum across 50 classes: {prob_sum:.6f} (Expected: 1.000000)")
        print(f"  • Top-5 Classes: {[(c, p) for c, p, _ in pred['top_5']]}")
        assert abs(prob_sum - 1.0) < 1e-4, f"Probabilities do not sum to 1: {prob_sum}"
        assert pred["inference_time_ms"] < 2000.0, f"Inference latency too slow: {pred['inference_time_ms']} ms"

        # 4. Visual Translation & Multi-mode Rendering
        t_vis = time.time()
        fig, params = translator.render_visual(
            y=y,
            sr=sr,
            predicted_class=pred["predicted_class"],
            confidence=pred["confidence_pct"],
            visual_mode="Class-Specific Generative Geometry",
            figsize=(6, 6),
            dpi=100
        )
        vis_latency = (time.time() - t_vis) * 1000.0
        print(f"  • Visual Render Time: {vis_latency:.2f} ms (Target: < 1000 ms)")
        print(f"  • Visual Parameters: Archetype='{params['archetype']}', Radius={params['base_radius']:.3f}, Vertices={params['vertices']}, Hue={params['hue_primary']}")
        assert vis_latency < 1000.0, f"Visual rendering too slow: {vis_latency} ms"

        # 5. Export to PNG
        saved_png = translator.save_visual_png(fig, predicted_class=pred["predicted_class"])
        assert Path(saved_png).exists() and Path(saved_png).stat().st_size > 0, f"Saved PNG invalid: {saved_png}"
        print(f"  • Exported Visual PNG: {Path(saved_png).name} ({Path(saved_png).stat().st_size / 1024:.1f} KB)")

        results.append({
            "sample": s["name"],
            "file": s["file"],
            "expected": s["expected_category"],
            "predicted": pred["predicted_class"],
            "confidence_pct": pred["confidence_pct"],
            "latency_ms": pred["inference_time_ms"],
            "archetype": params["archetype"],
            "exported_png": str(saved_png)
        })

    print("\n✓ All 3 test samples executed through the complete pipeline flawlessly!")
    return results


def test_gui_lifecycle():
    print("\n" + "=" * 70)
    print("STEP 3: Testing GUI Application Lifecycle & Tab Wiring")
    print("=" * 70)

    # Initialize app without blocking mainloop
    app = SoundVisualizerApp()
    app.update_idletasks()

    print("✓ SoundVisualizerApp created successfully.")
    print(f"  • Window Title: {app.title()}")
    print(f"  • Number of Tabs: {len(app.notebook.tabs())} (Expected: 5)")
    assert len(app.notebook.tabs()) == 5, f"Expected 5 tabs, got {len(app.notebook.tabs())}"

    # Verify Tab 1 (Home)
    app.refresh_model_status()
    badge_text = app.lbl_model_status_badge.cget("text")
    details_text = app.lbl_model_details.cget("text")
    print(f"  • Tab 1 (Home) Badge: {badge_text}")
    print(f"  • Tab 1 (Home) Details:\n{details_text}")
    assert "READY" in badge_text or "LOADED" in badge_text, f"Unexpected badge text: {badge_text}"

    # Verify Tab 5 (Performance)
    app.refresh_performance_tab()
    test_acc_text = app.perf_metric_labels["Test Accuracy"].cget("text")
    test_prec_text = app.perf_metric_labels["Test Macro Precision"].cget("text")
    test_rec_text = app.perf_metric_labels["Test Macro Recall"].cget("text")
    test_f1_text = app.perf_metric_labels["Test Macro F1-Score"].cget("text")
    print(f"  • Tab 5 (Performance) Metrics: Acc={test_acc_text}, Prec={test_prec_text}, Rec={test_rec_text}, F1={test_f1_text}")
    assert "71.0" in test_acc_text, f"Expected 71.00% accuracy, got {test_acc_text}"
    assert "70.62" in test_prec_text, f"Expected 70.62% precision, got {test_prec_text}"
    assert "69.24" in test_f1_text, f"Expected 69.24% F1, got {test_f1_text}"

    # Programmatic file load into GUI
    test_file = r"K:/AI project/Datasets/audio/1-100032-A-0.wav"
    y, sr = load_audio(test_file, target_sr=TARGET_SAMPLE_RATE)
    meta = get_audio_metadata(test_file, y, sr)
    app.current_audio_path = test_file
    app.current_audio_data = y
    app.current_sr = sr
    app.current_metadata = meta

    # Populate Tab 2 (Analysis)
    v_feats = extract_all_visual_features(y, sr)
    app.meta_stat_labels["Duration"].config(text=f"{meta['processed_duration_sec']:.2f} s")
    app.meta_stat_labels["Sample Rate"].config(text=f"{meta['processed_sample_rate']:,} Hz")
    app.meta_stat_labels["RMS Energy"].config(text=f"{meta['rms_energy']:.4f}")
    app.meta_stat_labels["Peak Amp"].config(text=f"{meta['peak_amplitude']:.4f}")
    app.meta_stat_labels["Centroid"].config(text=f"{v_feats['raw_centroid_hz']:.1f} Hz")
    app.meta_stat_labels["ZCR"].config(text=f"{v_feats['raw_zcr_mean']:.4f}")
    app.meta_stat_labels["Dominant Freq"].config(text=f"{v_feats['raw_dominant_freq_hz']:.1f} Hz")
    app.meta_stat_labels["Dominant Pitch"].config(text=f"{v_feats['dominant_pitch']}")
    app.update_audio_analysis_plots()
    print("  • Tab 2 (Analysis) Plots & 8 Stat Labels updated successfully.")

    # Programmatic Prediction
    pred = app.predictor.predict(test_file)
    app._on_prediction_success(pred)
    assert app.lbl_hero_class.cget("text") == "DOG", f"Hero class mismatch: {app.lbl_hero_class.cget('text')}"
    print(f"  • Tab 3 (Prediction) Hero Class: {app.lbl_hero_class.cget('text')} ({app.lbl_hero_conf.cget('text')})")

    # Programmatic Visual Translation
    app.action_generate_visual()
    assert app.current_visual_fig is not None, "Visual figure was not created!"
    print(f"  • Tab 4 (Visual Translation) Rendered: {app.telemetry_labels['Predicted Class'].cget('text')}")

    # Destroy window cleanly
    app.destroy()
    print("✓ SoundVisualizerApp closed cleanly after passing all lifecycle tests!")


if __name__ == "__main__":
    predictor = test_model_loading()
    results = test_end_to_end_samples(predictor)
    test_gui_lifecycle()
    print("\n" + "=" * 70)
    print("🎉 ALL TESTS PASSED SUCCESSFULLY! GUI & EXPERIMENT 4 FULLY INTEGRATED!")
    print("=" * 70)

