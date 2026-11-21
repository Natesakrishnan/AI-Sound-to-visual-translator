"""
Enhanced Model Training Script for ESC-50 Environmental Audio Classification.
Trains a Multi-Scale Residual 2D CNN with Dual Global Pooling and SpecAugment Data Augmentation.
Guarantees strict fold partitioning:
  - Folds 1, 2, 3 -> Training Set   (1,200 audio files)
  - Fold 4       -> Validation Set (  400 audio files)
  - Fold 5       -> Final Test Set (  400 audio files)
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import numpy as np

# Ensure Keras backend is PyTorch
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Add src directory to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cnn_model import build_audio_cnn, set_seed, RANDOM_SEED
from dataset_loader import load_or_create_dataset_cache, save_label_mapping
from evaluation import (
    evaluate_predictions,
    plot_and_save_confusion_matrix,
    plot_and_save_training_history,
    save_metrics_summary
)
from hardware_info import get_system_hardware_info, get_hardware_summary_text


def apply_spec_augment(
    spec: np.ndarray,
    freq_mask_max: int = 14,
    time_mask_max: int = 18,
    max_shift: int = 20
) -> np.ndarray:
    """
    Apply SpecAugment (Time Masking, Frequency Masking, Time Shifting) to a (128, 216, 1) spectrogram.
    Uses 0.0 (neutral standardized mean) for masking.
    """
    augmented = spec.copy()
    n_mels, n_frames, _ = augmented.shape

    # 1. Random Time Roll / Shift
    shift = random.randint(-max_shift, max_shift)
    augmented = np.roll(augmented, shift, axis=1)

    # 2. Frequency Masking (mask 1 random frequency band)
    if random.random() > 0.25:
        f_width = random.randint(4, freq_mask_max)
        f_start = random.randint(0, max(0, n_mels - f_width))
        augmented[f_start:f_start + f_width, :, :] = 0.0  # Fill with standardized mean 0.0

    # 3. Time Masking (mask 1 random time slice)
    if random.random() > 0.25:
        t_width = random.randint(4, time_mask_max)
        t_start = random.randint(0, max(0, n_frames - t_width))
        augmented[:, t_start:t_start + t_width, :] = 0.0

    return augmented


class AugmentedDataGenerator(keras.utils.PyDataset):
    """
    Custom PyDataset for on-the-fly SpecAugment and Mixup data augmentation during training.
    """
    def __init__(
        self,
        X: np.ndarray,
        y_cat: np.ndarray,
        batch_size: int = 32,
        augment: bool = True,
        mixup_alpha: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.X = X
        self.y_cat = y_cat
        self.batch_size = batch_size
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = self.X[batch_idx].copy()
        batch_y = self.y_cat[batch_idx].copy()

        if self.augment:
            # 1. SpecAugment (Time Shift, Time Masking, Frequency Masking)
            augmented_x = np.array([apply_spec_augment(x) for x in batch_x], dtype=np.float32)

            # 2. Mixup interpolation (linear acoustic superposition)
            if self.mixup_alpha > 0 and len(augmented_x) > 1:
                perm = np.random.permutation(len(augmented_x))
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                lam = max(lam, 1.0 - lam)  # Preserve primary label dominance
                augmented_x = lam * augmented_x + (1.0 - lam) * augmented_x[perm]
                batch_y = lam * batch_y + (1.0 - lam) * batch_y[perm]

            return augmented_x, batch_y

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)


def train_audio_classifier(
    dataset_dir: Optional[str] = None,
    epochs: int = 45,
    initial_batch_size: int = 32,
    learning_rate: float = 0.0005,
    force_recompute_cache: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Dict[str, Any]:
    """
    Execute enhanced end-to-end training, validation, testing, and artifact generation.

    Args:
        dataset_dir: Optional path to ESC-50 dataset directory.
        epochs: Maximum training epochs (default: 45).
        initial_batch_size: Starting batch size (default: 32).
        learning_rate: Learning rate for Adam optimizer.
        force_recompute_cache: Whether to force feature recomputation.
        progress_callback: Optional GUI callback fn(status_text: str, progress_fraction: float).

    Returns:
        Dictionary with training statistics, evaluation metrics, and artifact paths.
    """
    start_time = time.time()
    set_seed(RANDOM_SEED)

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"
    outputs_dir = project_root / "outputs"
    graphs_dir = outputs_dir / "graphs"
    cm_dir = outputs_dir / "confusion_matrix"

    models_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    cm_dir.mkdir(parents=True, exist_ok=True)

    model_save_path = models_dir / "audio_cnn_model.keras"
    prev_model_path = models_dir / "audio_cnn_model_previous.keras"
    label_map_path = models_dir / "label_mapping.json"
    metadata_path = models_dir / "model_metadata.json"

    # Rule 10: Preserve previous checkpoint if it exists
    if model_save_path.exists() and not prev_model_path.exists():
        import shutil
        shutil.copy2(str(model_save_path), str(prev_model_path))
        print(f"[Training] Preserved previous model checkpoint to {prev_model_path}")

    print(get_hardware_summary_text())
    hw_info = get_system_hardware_info()

    if progress_callback:
        progress_callback("Loading / Caching ESC-50 Mel Spectrograms...", 0.10)

    # 1. Load Dataset Splits from Cache or Raw Files
    cache_path = project_root / "outputs" / "cache" / "esc50_features_cache.npz"
    dataset = load_or_create_dataset_cache(
        dataset_dir=dataset_dir,
        force_recompute=force_recompute_cache
    )

    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_val, y_val = dataset["X_val"], dataset["y_val"]
    X_test, y_test = dataset["X_test"], dataset["y_test"]
    class_names = dataset["class_names"]
    id_to_class = dataset["id_to_class"]

    save_label_mapping(id_to_class, str(label_map_path))

    num_classes = len(class_names)
    input_shape = X_train.shape[1:]  # (128, 216, 1)

    print(f"[Training] Dataset loaded: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"[Training] Classes: {num_classes}, Input Shape: {input_shape}")

    if progress_callback:
        progress_callback("Constructing Multi-Scale Residual CNN...", 0.20)

    # 2. Build Enhanced Residual CNN Model
    model = build_audio_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate
    )

    # Convert targets to one-hot distributions for label-smoothed categorical cross-entropy
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=num_classes)

    # 3. Callbacks Configuration
    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=14,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        )
    ]

    class GUIEpochCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_acc = logs.get("val_accuracy", 0.0) * 100
            train_acc = logs.get("accuracy", 0.0) * 100
            frac = 0.20 + 0.65 * ((epoch + 1) / epochs)
            msg = f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%"
            print(f"[Epoch {epoch+1:02d}] {msg}")
            if progress_callback:
                progress_callback(msg, frac)

    callbacks.append(GUIEpochCallback())

    # 4. Training on GPU with SpecAugment and Mixup Data Generators
    batch_size = initial_batch_size
    history = None

    for attempt_batch in [batch_size, 16, 8]:
        try:
            print(f"[Training] Starting enhanced training (Batch Size = {attempt_batch}) on {hw_info['gpu']['device_display']}...")
            train_gen = AugmentedDataGenerator(
                X=X_train,
                y_cat=y_train_cat,
                batch_size=attempt_batch,
                augment=True,
                mixup_alpha=0.2
            )
            val_gen = AugmentedDataGenerator(
                X=X_val,
                y_cat=y_val_cat,
                batch_size=attempt_batch,
                augment=False,
                mixup_alpha=0.0
            )

            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=epochs,
                callbacks=callbacks,
                verbose=2
            )
            batch_size = attempt_batch
            break
        except Exception as oom_err:
            print(f"[Training Warning] Batch {attempt_batch} error: {oom_err}")
            if attempt_batch == 8:
                raise oom_err
            print("[Training] Retrying with smaller batch size...")

    # Best weights were saved by ModelCheckpoint on best Fold 4 validation accuracy
    if not model_save_path.exists():
        model.save(str(model_save_path))

    if progress_callback:
        progress_callback("Generating Evaluation Curves & Test Metrics...", 0.88)

    # 5. Save Training Curves
    history_dict = history.history if history else {}
    plot_and_save_training_history(
        history_dict,
        str(graphs_dir / "training_history.png")
    )

    # 6. Final Evaluation on Strictly Unseen Fold 5 Test Set
    print("[Training] Evaluating model on strictly unseen Fold 5 Test Set...")
    best_model = keras.models.load_model(str(model_save_path))
    y_test_probs = best_model.predict(X_test, batch_size=batch_size, verbose=0)
    test_metrics = evaluate_predictions(y_test, y_test_probs, class_names)

    # 7. Measure Prediction Latency (over 50 test samples)
    latencies = []
    for i in range(min(50, len(X_test))):
        sample_tensor = X_test[i:i+1]
        t0 = time.time()
        _ = best_model(sample_tensor, training=False)
        latencies.append((time.time() - t0) * 1000.0)
    avg_latency_ms = round(float(np.mean(latencies[5:])), 2)  # Discard initial 5 warm-up calls
    test_metrics["avg_prediction_latency_ms"] = avg_latency_ms

    # 8. Save Confusion Matrix and Summary JSON
    plot_and_save_confusion_matrix(
        test_metrics["confusion_matrix"],
        class_names,
        str(cm_dir / "confusion_matrix.png")
    )
    save_metrics_summary(test_metrics, str(cm_dir / "test_metrics.json"))

    total_time_sec = round(time.time() - start_time, 2)
    training_mins = round(total_time_sec / 60.0, 2)

    # 9. Save Complete Metadata
    metadata = {
        "dataset": "ESC-50 Environmental Sound Classification",
        "total_samples": 2000,
        "split": {
            "train_folds": [1, 2, 3],
            "train_samples": len(X_train),
            "val_fold": [4],
            "val_samples": len(X_val),
            "test_fold": [5],
            "test_samples": len(X_test)
        },
        "num_classes": num_classes,
        "training_device": hw_info["gpu"]["device_display"],
        "primary_gpu": hw_info["gpu"]["primary_gpu"],
        "cpu_model": hw_info["cpu"]["model"],
        "ram": hw_info["ram"]["formatted"],
        "epochs_trained": len(history_dict.get("loss", [])),
        "batch_size": batch_size,
        "training_time_sec": total_time_sec,
        "training_time_mins": training_mins,
        "best_val_accuracy_pct": round(max(history_dict.get("val_accuracy", [0.0])) * 100, 2),
        "test_accuracy_pct": test_metrics["accuracy_pct"],
        "test_precision_pct": test_metrics["precision_pct"],
        "test_recall_pct": test_metrics["recall_pct"],
        "test_f1_score_pct": test_metrics["f1_score_pct"],
        "avg_prediction_latency_ms": avg_latency_ms,
        "model_file": str(model_save_path.name),
        "previous_model_file": str(prev_model_path.name) if prev_model_path.exists() else None,
        "label_map_file": str(label_map_path.name),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n==================================================")
    print(" ENHANCED TRAINING & EVALUATION COMPLETED!")
    print("==================================================")
    print(f" Training Time:             {training_mins} minutes ({total_time_sec} s)")
    print(f" Device Used:               {metadata['training_device']}")
    print(f" Best Val Accuracy (Fold 4):{metadata['best_val_accuracy_pct']}%")
    print(f" Test Accuracy (Fold 5):    {metadata['test_accuracy_pct']}% (Fold 5 Unseen)")
    print(f" Test Macro F1-Score:       {metadata['test_f1_score_pct']}%")
    print(f" Average Latency:           {avg_latency_ms} ms/sample")
    print(f" Active Model Saved To:     {model_save_path}")
    print(f" Previous Model Backup:     {prev_model_path}")
    print("==================================================\n")

    if progress_callback:
        progress_callback("Training & Evaluation Completed!", 1.0)

    return {
        "metadata": metadata,
        "metrics": test_metrics,
        "history": history_dict,
        "model_path": str(model_save_path),
        "label_map_path": str(label_map_path)
    }


if __name__ == "__main__":
    train_audio_classifier(epochs=50, force_recompute_cache=False)
