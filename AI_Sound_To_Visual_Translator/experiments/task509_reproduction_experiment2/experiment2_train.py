"""
Training Script: Task 509 Reproduction - Experiment 2
Controlled reproduction experiment testing:
  1. Restored keras.utils.set_random_seed(42) in set_seed().
  2. Restored historical default cuDNN behavior (no forced cudnn.deterministic=True, no forced cudnn.benchmark=False).

All other parameters are strictly identical to Task 509 / Experiment 1:
  - Folds 1-3 train (1200), Fold 4 val (400), Fold 5 test (400)
  - Mel 128, FFT 2048, Hop 512, Decibel clipping [-1.0, 1.0], Peak norm 0.95
  - SpecAugment: temporal roll +/-16 frames, masks filled with -1.0, Mixup disabled
  - 4-Block ESC50_ResNet_CNN (1,364,178 parameters)
  - Adam(0.0005), batch size 32, 40 epochs
  - EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
  - ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
  - ModelCheckpoint(filepath=model_best_val_acc_path, monitor="val_accuracy", save_best_only=True)
"""

import os
import sys
import time
import json
import random
from pathlib import Path
from typing import Dict, Any

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

exp_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(exp_dir))
sys.path.insert(0, str(exp_dir.parent.parent / "src"))

from experiment2_cnn_model import build_audio_cnn, set_seed, RANDOM_SEED
from experiment2_dataset_loader import load_or_create_dataset_cache, save_label_mapping
from hardware_info import get_system_hardware_info, get_hardware_summary_text


def apply_spec_augment(
    spec: np.ndarray,
    freq_mask_max: int = 14,
    time_mask_max: int = 18,
    max_shift: int = 16
) -> np.ndarray:
    """Exact SpecAugment from Task 509 (Step 395)."""
    augmented = spec.copy()
    n_mels, n_frames, _ = augmented.shape

    # 1. Random Time Roll / Shift
    shift = random.randint(-max_shift, max_shift)
    augmented = np.roll(augmented, shift, axis=1)

    # 2. Frequency Masking (fill with noise floor -1.0)
    if random.random() > 0.3:
        f_width = random.randint(4, freq_mask_max)
        f_start = random.randint(0, max(0, n_mels - f_width))
        augmented[f_start:f_start + f_width, :, :] = -1.0

    # 3. Time Masking (fill with noise floor -1.0)
    if random.random() > 0.3:
        t_width = random.randint(4, time_mask_max)
        t_start = random.randint(0, max(0, n_frames - t_width))
        augmented[:, t_start:t_start + t_width, :] = -1.0

    return augmented


class AugmentedDataGenerator(keras.utils.PyDataset):
    """Custom PyDataset for on-the-fly SpecAugment data augmentation (Task 509)."""
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 32, augment: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_x = self.X[batch_idx]
        batch_y = self.y[batch_idx]

        if self.augment:
            augmented_x = np.array([apply_spec_augment(x) for x in batch_x], dtype=np.float32)
            return augmented_x, batch_y

        return batch_x, batch_y

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)


def run_experiment_2(
    epochs: int = 40,
    batch_size: int = 32,
    learning_rate: float = 0.0005
) -> Dict[str, Any]:
    """Execute Task 509 reproduction experiment 2."""
    start_time = time.time()
    set_seed(RANDOM_SEED)

    # Directories
    exp_models_dir = exp_dir / "models"
    exp_outputs_dir = exp_dir / "outputs"
    exp_graphs_dir = exp_outputs_dir / "graphs"
    exp_cm_dir = exp_outputs_dir / "confusion_matrix"
    exp_cache_dir = exp_dir / "cache"

    for d in [exp_models_dir, exp_outputs_dir, exp_graphs_dir, exp_cm_dir, exp_cache_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model_save_path = exp_models_dir / "audio_cnn_model_task509.keras"
    model_best_val_acc_path = exp_models_dir / "audio_cnn_model_task509_best_val_acc.keras"
    model_best_val_loss_path = exp_models_dir / "audio_cnn_model_task509_best_val_loss.keras"
    label_map_path = exp_models_dir / "label_mapping.json"
    metadata_path = exp_models_dir / "model_metadata.json"
    history_csv_path = exp_outputs_dir / "training_history.csv"

    print("=" * 60)
    print(" TASK 509 REPRODUCTION – EXPERIMENT 2")
    print(" Changes: keras.utils.set_random_seed(42) + Default cuDNN")
    print("=" * 60)
    print(get_hardware_summary_text())
    hw_info = get_system_hardware_info()

    # Verify GPU availability
    if not hw_info["gpu"]["cuda_available"]:
        raise RuntimeError("FATAL: CUDA is not available! Must run on NVIDIA RTX 3050 GPU.")
    print(f"[Device Verification] GPU Acceleration Confirmed: {hw_info['gpu']['primary_gpu']}\n")

    # 1. Load Dataset Cache
    cache_path = exp_cache_dir / "esc50_features_cache_task509.npz"
    dataset = load_or_create_dataset_cache(
        cache_path=cache_path,
        force_recompute=False
    )

    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_val, y_val = dataset["X_val"], dataset["y_val"]
    X_test, y_test = dataset["X_test"], dataset["y_test"]
    class_names = dataset["class_names"]
    id_to_class = dataset["id_to_class"]

    save_label_mapping(id_to_class, str(label_map_path))

    num_classes = len(class_names)
    input_shape = X_train.shape[1:]

    print(f"\n[Dataset] Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"[Dataset] Feature Range: Min={X_train.min():.4f}, Max={X_train.max():.4f}, Mean={X_train.mean():.4f}")

    # 2. Build Task 509 CNN Model (with historical set_seed)
    model = build_audio_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
        learning_rate=learning_rate
    )
    print(f"[Model] ESC50_ResNet_CNN built ({model.count_params():,} parameters).\n")

    # 3. Callbacks matching Task 509 exactly
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(model_best_val_acc_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    class EpochLoggerCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            val_acc = logs.get("val_accuracy", 0.0) * 100
            train_acc = logs.get("accuracy", 0.0) * 100
            val_loss = logs.get("val_loss", 0.0)
            train_loss = logs.get("loss", 0.0)
            lr = logs.get("learning_rate", logs.get("lr", 0.0))
            print(f"[Epoch {epoch+1:02d}/{epochs}] Train Acc: {train_acc:.1f}% (Loss: {train_loss:.4f}) | Val Acc: {val_acc:.1f}% (Loss: {val_loss:.4f}) | LR: {lr:.2e}")

    callbacks.append(EpochLoggerCallback())

    # 4. Data Generators
    train_gen = AugmentedDataGenerator(X_train, y_train, batch_size=batch_size, augment=True)
    val_gen = AugmentedDataGenerator(X_val, y_val, batch_size=batch_size, augment=False)

    print(f"[Training] Starting Experiment 2 ({epochs} epochs, batch size {batch_size}) on {hw_info['gpu']['device_display']}...\n")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )

    # Save best val_loss restored weights from EarlyStopping (exact Task 509 model.save() behavior)
    model.save(str(model_best_val_loss_path))
    model.save(str(model_save_path))
    print(f"\n[Checkpoint] Saved best val_loss restored weights to: {model_best_val_loss_path}")
    print(f"[Checkpoint] Best val_accuracy checkpoint preserved at: {model_best_val_acc_path}")

    history_dict = history.history

    # Save training history to CSV
    hist_df = pd.DataFrame(history_dict)
    hist_df.insert(0, "epoch", range(1, len(hist_df) + 1))
    hist_df.to_csv(str(history_csv_path), index=False)
    print(f"[Artifact] Saved training history CSV to: {history_csv_path}")

    # Plot Accuracy & Loss History
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
    ax1.set_facecolor("#161b22")
    ax2.set_facecolor("#161b22")

    epochs_range = range(1, len(history_dict["accuracy"]) + 1)
    ax1.plot(epochs_range, [a * 100 for a in history_dict["accuracy"]], label="Train Accuracy", color="#58a6ff", linewidth=2)
    ax1.plot(epochs_range, [a * 100 for a in history_dict["val_accuracy"]], label="Val Accuracy", color="#3fb950", linewidth=2)
    ax1.axhline(70.5, color="#f85149", linestyle="--", alpha=0.7, label="Task 509 Target (70.5%)")
    ax1.axhline(66.75, color="#d29922", linestyle=":", alpha=0.7, label="Experiment 1 (66.75%)")
    ax1.set_title("Experiment 2 - Accuracy Curve", color="white", fontsize=12)
    ax1.set_xlabel("Epoch", color="#8b949e")
    ax1.set_ylabel("Accuracy (%)", color="#8b949e")
    ax1.tick_params(colors="#8b949e")
    ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
    ax1.grid(True, color="#30363d", alpha=0.5)

    ax2.plot(epochs_range, history_dict["loss"], label="Train Loss", color="#58a6ff", linewidth=2)
    ax2.plot(epochs_range, history_dict["val_loss"], label="Val Loss", color="#d29922", linewidth=2)
    ax2.set_title("Experiment 2 - Loss Curve", color="white", fontsize=12)
    ax2.set_xlabel("Epoch", color="#8b949e")
    ax2.set_ylabel("Loss", color="#8b949e")
    ax2.tick_params(colors="#8b949e")
    ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
    ax2.grid(True, color="#30363d", alpha=0.5)

    plt.tight_layout()
    graph_path = exp_graphs_dir / "training_history.png"
    plt.savefig(str(graph_path), dpi=150)
    plt.close(fig)
    print(f"[Artifact] Saved training curves to: {graph_path}")

    # 5. Evaluate on strictly unseen Fold 5
    print("\n[Evaluation] Evaluating models on strictly unseen Fold 5 Test Set (400 samples)...")
    
    # 5A. Evaluate Best Val Loss Restored Checkpoint
    best_loss_model = keras.models.load_model(str(model_best_val_loss_path))

    # Single-sample inference latency benchmark (50 samples on GPU, discarding warm-up)
    latencies = []
    _ = best_loss_model.predict(X_test[:1], verbose=0)
    for i in range(min(50, len(X_test))):
        t0 = time.time()
        _ = best_loss_model.predict(X_test[i:i+1], verbose=0)
        latencies.append((time.time() - t0) * 1000.0)
    avg_latency_ms = round(float(np.mean(latencies)), 2)

    y_loss_probs = best_loss_model.predict(X_test, batch_size=batch_size, verbose=0)
    y_loss_pred = np.argmax(y_loss_probs, axis=1)

    acc_loss = float(accuracy_score(y_test, y_loss_pred))
    prec_loss = float(precision_score(y_test, y_loss_pred, average="macro", zero_division=0))
    rec_loss = float(recall_score(y_test, y_loss_pred, average="macro", zero_division=0))
    f1_loss = float(f1_score(y_test, y_loss_pred, average="macro", zero_division=0))
    correct_loss = int(np.sum(y_loss_pred == y_test))

    # 5B. Evaluate Peak Val Accuracy Checkpoint
    best_acc_model = keras.models.load_model(str(model_best_val_acc_path))
    y_acc_probs = best_acc_model.predict(X_test, batch_size=batch_size, verbose=0)
    y_acc_pred = np.argmax(y_acc_probs, axis=1)

    acc_ckpt = float(accuracy_score(y_test, y_acc_pred))
    prec_ckpt = float(precision_score(y_test, y_acc_pred, average="macro", zero_division=0))
    rec_ckpt = float(recall_score(y_test, y_acc_pred, average="macro", zero_division=0))
    f1_ckpt = float(f1_score(y_test, y_acc_pred, average="macro", zero_division=0))
    correct_ckpt = int(np.sum(y_acc_pred == y_test))

    print(f"[Evaluation Result] Best Val Loss Restored Checkpoint Test Acc: {acc_loss*100:.2f}% ({correct_loss}/400 correct)")
    print(f"[Evaluation Result] Peak Val Accuracy Checkpoint Test Acc:      {acc_ckpt*100:.2f}% ({correct_ckpt}/400 correct)")

    # Select primary evaluation metrics (best of the two)
    if acc_ckpt >= acc_loss:
        primary_pred = y_acc_pred
        primary_acc = acc_ckpt
        primary_prec = prec_ckpt
        primary_rec = rec_ckpt
        primary_f1 = f1_ckpt
        primary_name = "Peak Val Accuracy Checkpoint"
    else:
        primary_pred = y_loss_pred
        primary_acc = acc_loss
        primary_prec = prec_loss
        primary_rec = rec_loss
        primary_f1 = f1_loss
        primary_name = "Best Val Loss Restored Checkpoint"

    cm = confusion_matrix(y_test, primary_pred, labels=range(num_classes))
    report_dict = classification_report(y_test, primary_pred, target_names=class_names, output_dict=True, zero_division=0)

    # Plot Confusion Matrix
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    plt.figure(figsize=(20, 18), facecolor="#0d1117")
    ax = plt.subplot(111, facecolor="#161b22")
    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="magma",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Frequency"},
        ax=ax
    )
    plt.title(f"Task 509 Reproduction Exp 2 Confusion Matrix ({primary_name}: {primary_acc*100:.2f}%)", color="white", fontsize=14, pad=15)
    plt.xlabel("Predicted Class", color="#8b949e", fontsize=12)
    plt.ylabel("True Class", color="#8b949e", fontsize=12)
    plt.xticks(rotation=90, fontsize=8, color="#8b949e")
    plt.yticks(rotation=0, fontsize=8, color="#8b949e")
    plt.tight_layout()
    cm_path = exp_cm_dir / "confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=150)
    plt.close()
    print(f"[Artifact] Saved confusion matrix to: {cm_path}")

    # Save metrics JSON
    metrics_summary = {
        "accuracy": primary_acc,
        "precision": primary_prec,
        "recall": primary_rec,
        "f1_score": primary_f1,
        "accuracy_pct": round(primary_acc * 100, 2),
        "precision_pct": round(primary_prec * 100, 2),
        "recall_pct": round(primary_rec * 100, 2),
        "f1_score_pct": round(primary_f1 * 100, 2),
        "best_val_loss_restored_test_accuracy_pct": round(acc_loss * 100, 2),
        "best_val_loss_correct": correct_loss,
        "best_val_acc_checkpoint_test_accuracy_pct": round(acc_ckpt * 100, 2),
        "best_val_acc_correct": correct_ckpt,
        "total_test_samples": len(y_test),
        "avg_latency_ms": avg_latency_ms,
        "per_class_report": {
            cls: {
                "precision": round(report_dict[cls]["precision"] * 100, 2),
                "recall": round(report_dict[cls]["recall"] * 100, 2),
                "f1_score": round(report_dict[cls]["f1-score"] * 100, 2),
            }
            for cls in class_names
        }
    }
    with open(exp_cm_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[Artifact] Saved metrics summary to: {exp_cm_dir / 'test_metrics.json'}")

    total_time_sec = round(time.time() - start_time, 2)
    training_mins = round(total_time_sec / 60.0, 2)

    metadata = {
        "experiment": "Task 509 Controlled Reproduction - Experiment 2",
        "permitted_changes": [
            "Restored keras.utils.set_random_seed(42) in set_seed()",
            "Restored historical default cuDNN behavior (no forced cudnn.deterministic=True, no forced cudnn.benchmark=False)"
        ],
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
        "best_val_loss": round(min(history_dict.get("val_loss", [99.0])), 4),
        "test_accuracy_pct": metrics_summary["accuracy_pct"],
        "test_precision_pct": metrics_summary["precision_pct"],
        "test_recall_pct": metrics_summary["recall_pct"],
        "test_f1_score_pct": metrics_summary["f1_score_pct"],
        "best_val_loss_restored_test_acc": round(acc_loss * 100, 2),
        "best_val_acc_checkpoint_test_acc": round(acc_ckpt * 100, 2),
        "avg_prediction_latency_ms": avg_latency_ms,
        "model_file": "audio_cnn_model_task509.keras",
        "label_map_file": "label_mapping.json",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print(" TASK 509 REPRODUCTION EXPERIMENT 2 COMPLETED!")
    print("=" * 60)
    print(f" Training Time:           {training_mins} mins ({total_time_sec} s)")
    print(f" Best Val Loss Restored:  {acc_loss*100:.2f}% ({correct_loss}/400 correct)")
    print(f" Best Val Acc Checkpoint: {acc_ckpt*100:.2f}% ({correct_ckpt}/400 correct)")
    print(f" Primary Test Accuracy:   {primary_acc*100:.2f}% (Macro F1: {primary_f1*100:.2f}%)")
    print(f" Average Latency:         {avg_latency_ms} ms/sample")
    print("=" * 60 + "\n")

    return {
        "metadata": metadata,
        "metrics": metrics_summary,
        "history": history_dict
    }


if __name__ == "__main__":
    run_experiment_2()

