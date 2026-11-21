"""
Model Evaluation and Performance Visualization Module
Calculates Accuracy, Precision, Recall, F1-Score, generates 50-Class Confusion Matrix,
and plots Training/Validation History curves.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless rendering for saving plots safely
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics from ground truth and predicted probabilities.

    Returns:
        Dictionary containing overall accuracy, macro precision, macro recall,
        macro F1 score, confusion matrix array, and classification report dict.
    """
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": round(acc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "accuracy_pct": round(acc * 100, 2),
        "precision_pct": round(float(prec) * 100, 2),
        "recall_pct": round(float(rec) * 100, 2),
        "f1_score_pct": round(float(f1) * 100, 2),
        "confusion_matrix": cm,
        "classification_report": report_dict,
        "total_test_samples": len(y_true),
        "correct_predictions": int(np.sum(y_true == y_pred))
    }


def plot_and_save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str
):
    """
    Generate and save a clear, high-resolution 50-class confusion matrix heatmap.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(18, 16), dpi=150)
    plt.style.use("dark_background")

    plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title("ESC-50 Sound Classification — 50-Class Confusion Matrix (Fold 5 Unseen Test)", fontsize=16, pad=20, weight="bold")
    plt.colorbar(fraction=0.046, pad=0.04, label="Sample Count")

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90, fontsize=8)
    plt.yticks(tick_marks, class_names, fontsize=8)

    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.grid(False)
    plt.tight_layout()

    plt.savefig(str(out_p), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Evaluation] Saved confusion matrix to: {out_p}")


def plot_and_save_training_history(
    history_dict: Dict[str, List[float]],
    output_path: str
):
    """
    Generate side-by-side Loss and Accuracy curves for Training vs Validation.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    fig.patch.set_facecolor("#0d1117")
    ax1.set_facecolor("#161b22")
    ax2.set_facecolor("#161b22")

    # Accuracy Plot
    train_acc = [a * 100 for a in history_dict.get("accuracy", [])]
    val_acc = [a * 100 for a in history_dict.get("val_accuracy", [])]
    ax1.plot(epochs, train_acc, "o-", color="#58a6ff", linewidth=2, label="Train Accuracy (Folds 1-3)")
    ax1.plot(epochs, val_acc, "s-", color="#3fb950", linewidth=2, label="Val Accuracy (Fold 4)")
    ax1.set_title("Training & Validation Accuracy", color="white", fontsize=13, weight="bold")
    ax1.set_xlabel("Epoch", color="#8b949e", fontsize=11)
    ax1.set_ylabel("Accuracy (%)", color="#8b949e", fontsize=11)
    ax1.tick_params(colors="#c9d1d9")
    ax1.grid(True, color="#30363d", linestyle="--", alpha=0.6)
    ax1.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    # Loss Plot
    train_loss = history_dict.get("loss", [])
    val_loss = history_dict.get("val_loss", [])
    ax2.plot(epochs, train_loss, "o-", color="#f85149", linewidth=2, label="Train Loss (Folds 1-3)")
    ax2.plot(epochs, val_loss, "s-", color="#d29922", linewidth=2, label="Val Loss (Fold 4)")
    ax2.set_title("Training & Validation Loss", color="white", fontsize=13, weight="bold")
    ax2.set_xlabel("Epoch", color="#8b949e", fontsize=11)
    ax2.set_ylabel("Categorical Cross-Entropy Loss", color="#8b949e", fontsize=11)
    ax2.tick_params(colors="#c9d1d9")
    ax2.grid(True, color="#30363d", linestyle="--", alpha=0.6)
    ax2.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="white")

    plt.tight_layout()
    plt.savefig(str(out_p), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Evaluation] Saved training history graphs to: {out_p}")


def save_metrics_summary(metrics: Dict[str, Any], output_path: str):
    """Save test evaluation metrics dictionary to JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "accuracy_pct": metrics["accuracy_pct"],
        "precision_pct": metrics["precision_pct"],
        "recall_pct": metrics["recall_pct"],
        "f1_score_pct": metrics["f1_score_pct"],
        "total_test_samples": metrics["total_test_samples"],
        "correct_predictions": metrics["correct_predictions"]
    }
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"[Evaluation] Saved metrics summary to: {out_p}")
