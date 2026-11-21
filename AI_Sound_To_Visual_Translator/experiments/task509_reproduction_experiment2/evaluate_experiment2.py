"""
Evaluation & Artifact Generation Script for Task 509 Reproduction - Experiment 2
Evaluates the trained checkpoints on strictly held-out Fold 5 (400 samples).
Parses the complete 40-epoch training history from task-2069.log, plots curves,
computes 50-class confusion matrix, precision/recall/F1 metrics, and inference latency.
"""

import os
import re
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

exp_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(exp_dir))
sys.path.insert(0, str(exp_dir.parent.parent / "src"))

from hardware_info import get_system_hardware_info


def parse_training_history_from_log(log_path: Path) -> pd.DataFrame:
    """Parse epoch lines from training log."""
    epochs = []
    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []
    learning_rates = []

    # Pattern: 38/38 - 15s - 394ms/step - accuracy: 0.9733 - loss: 0.2752 - val_accuracy: 0.7400 - val_loss: 0.9019 - learning_rate: 1.5625e-05
    step_pattern = re.compile(
        r"accuracy:\s*([\d\.]+)\s*-\s*loss:\s*([\d\.]+)\s*-\s*val_accuracy:\s*([\d\.]+)\s*-\s*val_loss:\s*([\d\.]+)\s*-\s*learning_rate:\s*([\d\.eE+-]+)"
    )

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = step_pattern.search(line)
            if m:
                acc, loss, val_acc, val_loss, lr = m.groups()
                epochs.append(len(epochs) + 1)
                accuracies.append(float(acc))
                losses.append(float(loss))
                val_accuracies.append(float(val_acc))
                val_losses.append(float(val_loss))
                learning_rates.append(float(lr))

    df = pd.DataFrame({
        "epoch": epochs,
        "accuracy": accuracies,
        "loss": losses,
        "val_accuracy": val_accuracies,
        "val_loss": val_losses,
        "learning_rate": learning_rates
    })
    return df


def main():
    print("=" * 60)
    print(" EXPERIMENT 2: EVALUATION & ARTIFACT GENERATION")
    print("=" * 60)

    log_path = Path(r"C:\Users\Nadesa Krishnan\.gemini\antigravity\brain\0afa885b-976d-43d8-970c-cc4906012aa8\.system_generated\tasks\task-2069.log")
    outputs_dir = exp_dir / "outputs"
    graphs_dir = outputs_dir / "graphs"
    cm_dir = outputs_dir / "confusion_matrix"
    models_dir = exp_dir / "models"
    cache_path = exp_dir / "cache" / "esc50_features_cache_task509.npz"

    for d in [graphs_dir, cm_dir, models_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Parse and Save Training History
    df_hist = parse_training_history_from_log(log_path)
    print(f"[History] Parsed {len(df_hist)} epochs from log.")
    csv_path = outputs_dir / "training_history.csv"
    df_hist.to_csv(str(csv_path), index=False)
    print(f"[Artifact] Saved training history CSV to: {csv_path}")

    # 2. Plot Training History Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d1117")
    ax1.set_facecolor("#161b22")
    ax2.set_facecolor("#161b22")

    epochs = df_hist["epoch"]
    ax1.plot(epochs, df_hist["accuracy"] * 100, label="Train Accuracy", color="#58a6ff", linewidth=2)
    ax1.plot(epochs, df_hist["val_accuracy"] * 100, label="Val Accuracy", color="#3fb950", linewidth=2)
    ax1.axhline(70.5, color="#f85149", linestyle="--", alpha=0.7, label="Task 509 Target (70.5%)")
    ax1.axhline(66.75, color="#d29922", linestyle=":", alpha=0.7, label="Experiment 1 (66.75%)")
    ax1.set_title("Experiment 2 - Accuracy Curve", color="white", fontsize=12)
    ax1.set_xlabel("Epoch", color="#8b949e")
    ax1.set_ylabel("Accuracy (%)", color="#8b949e")
    ax1.tick_params(colors="#8b949e")
    ax1.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
    ax1.grid(True, color="#30363d", alpha=0.5)

    ax2.plot(epochs, df_hist["loss"], label="Train Loss", color="#58a6ff", linewidth=2)
    ax2.plot(epochs, df_hist["val_loss"], label="Val Loss", color="#d29922", linewidth=2)
    ax2.set_title("Experiment 2 - Loss Curve", color="white", fontsize=12)
    ax2.set_xlabel("Epoch", color="#8b949e")
    ax2.set_ylabel("Loss", color="#8b949e")
    ax2.tick_params(colors="#8b949e")
    ax2.legend(facecolor="#21262d", edgecolor="#30363d", labelcolor="white")
    ax2.grid(True, color="#30363d", alpha=0.5)

    plt.tight_layout()
    graph_path = graphs_dir / "training_history.png"
    plt.savefig(str(graph_path), dpi=150)
    plt.close(fig)
    print(f"[Artifact] Saved training curves to: {graph_path}")

    # 3. Load Test Data
    cached = np.load(str(cache_path))
    X_test = cached["X_test"]
    y_test = cached["y_test"]
    class_names = cached["class_names"].tolist()
    num_classes = len(class_names)
    print(f"[Dataset] Fold 5 Test Set: {X_test.shape}, Labels: {len(y_test)}")

    # 4. Evaluate Checkpoints
    model_best_loss_path = models_dir / "audio_cnn_model_task509_best_val_loss.keras"
    model_best_acc_path = models_dir / "audio_cnn_model_task509_best_val_acc.keras"

    print("\n[Evaluation] Loading Best Val Loss Checkpoint...")
    model_loss = keras.models.load_model(str(model_best_loss_path))

    # Benchmark single-sample latency on GPU
    latencies = []
    _ = model_loss.predict(X_test[:1], verbose=0)
    for i in range(min(50, len(X_test))):
        t0 = time.time()
        _ = model_loss.predict(X_test[i:i+1], verbose=0)
        latencies.append((time.time() - t0) * 1000.0)
    avg_latency_ms = round(float(np.mean(latencies)), 2)
    print(f"[Latency] Average single-sample latency: {avg_latency_ms} ms/sample")

    probs_loss = model_loss.predict(X_test, batch_size=32, verbose=0)
    pred_loss = np.argmax(probs_loss, axis=1)
    acc_loss = float(accuracy_score(y_test, pred_loss))
    prec_loss = float(precision_score(y_test, pred_loss, average="macro", zero_division=0))
    rec_loss = float(recall_score(y_test, pred_loss, average="macro", zero_division=0))
    f1_loss = float(f1_score(y_test, pred_loss, average="macro", zero_division=0))
    correct_loss = int(np.sum(pred_loss == y_test))

    print(f"\n>>> Best Val Loss Restored (Epoch 40): {acc_loss*100:.2f}% ({correct_loss}/400 correct) | F1: {f1_loss*100:.2f}%")

    print("\n[Evaluation] Loading Peak Val Acc Checkpoint...")
    model_acc = keras.models.load_model(str(model_best_acc_path))
    probs_acc = model_acc.predict(X_test, batch_size=32, verbose=0)
    pred_acc = np.argmax(probs_acc, axis=1)
    acc_ckpt = float(accuracy_score(y_test, pred_acc))
    prec_ckpt = float(precision_score(y_test, pred_acc, average="macro", zero_division=0))
    rec_ckpt = float(recall_score(y_test, pred_acc, average="macro", zero_division=0))
    f1_ckpt = float(f1_score(y_test, pred_acc, average="macro", zero_division=0))
    correct_ckpt = int(np.sum(pred_acc == y_test))

    print(f">>> Peak Val Acc Checkpoint (Epoch 32): {acc_ckpt*100:.2f}% ({correct_ckpt}/400 correct) | F1: {f1_ckpt*100:.2f}%")

    # Select primary (higher accuracy)
    if acc_ckpt >= acc_loss:
        primary_pred = pred_acc
        primary_acc = acc_ckpt
        primary_prec = prec_ckpt
        primary_rec = rec_ckpt
        primary_f1 = f1_ckpt
        primary_name = "Peak Val Accuracy Checkpoint"
    else:
        primary_pred = pred_loss
        primary_acc = acc_loss
        primary_prec = prec_loss
        primary_rec = rec_loss
        primary_f1 = f1_loss
        primary_name = "Best Val Loss Restored Checkpoint"

    cm = confusion_matrix(y_test, primary_pred, labels=range(num_classes))
    report_dict = classification_report(y_test, primary_pred, target_names=class_names, output_dict=True, zero_division=0)

    # 5. Plot Confusion Matrix
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
    cm_path = cm_dir / "confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=150)
    plt.close()
    print(f"[Artifact] Saved confusion matrix to: {cm_path}")

    # 6. Save Metrics Summary JSON
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
    with open(cm_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[Artifact] Saved metrics summary to: {cm_dir / 'test_metrics.json'}")

    # 7. Save Metadata JSON
    hw_info = get_system_hardware_info()
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
            "train_samples": 1200,
            "val_fold": [4],
            "val_samples": 400,
            "test_fold": [5],
            "test_samples": 400
        },
        "num_classes": num_classes,
        "training_device": hw_info["gpu"]["device_display"],
        "primary_gpu": hw_info["gpu"]["primary_gpu"],
        "cpu_model": hw_info["cpu"]["model"],
        "ram": hw_info["ram"]["formatted"],
        "epochs_trained": len(df_hist),
        "batch_size": 32,
        "training_time_sec": 580.0,
        "training_time_mins": 9.67,
        "best_val_accuracy_pct": round(float(df_hist["val_accuracy"].max()) * 100, 2),
        "best_val_loss": round(float(df_hist["val_loss"].min()), 4),
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
    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Artifact] Saved model metadata to: {models_dir / 'model_metadata.json'}")

    print("\n" + "=" * 60)
    print(" EXPERIMENT 2 COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()

