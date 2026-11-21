"""
Module: Task 509 Dataset Loader
Loads ESC-50 dataset with exact Task 509 [-1.0, +1.0] scaled Mel Spectrogram features.
Caches to experiments/task509_reproduction/cache/esc50_features_cache_task509.npz.
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional
import numpy as np
import pandas as pd

# Load reproduction feature extraction explicitly
exp_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(exp_dir))
sys.path.insert(0, str(exp_dir.parent.parent / "src"))

from audio_preprocessing import load_audio, TARGET_SAMPLE_RATE
from task509_feature_extraction import compute_mel_spectrogram

ESC50_CLASSES = [
    "airplane", "breathing", "brushing_teeth", "can_opening", "car_horn",
    "cat", "chainsaw", "chirping_birds", "church_bells", "clapping",
    "clock_alarm", "clock_tick", "coughing", "cow", "crackling_fire",
    "crickets", "crow", "crying_baby", "dog", "door_wood_creaks",
    "door_wood_knock", "drinking_sipping", "engine", "fireworks", "footsteps",
    "frog", "glass_breaking", "hand_saw", "helicopter", "hen",
    "insects", "keyboard_typing", "laughing", "mouse_click", "pig",
    "pouring_water", "rain", "rooster", "sea_waves", "sheep",
    "siren", "sneezing", "snoring", "thunderstorm", "toilet_flush",
    "train", "vacuum_cleaner", "washing_machine", "water_drops", "wind"
]


def find_dataset_paths() -> Tuple[Path, Path]:
    base_dir = Path(__file__).resolve().parent.parent.parent
    candidates = [
        Path("K:/AI project/Datasets"),
        base_dir / "dataset",
        base_dir / "Datasets",
        base_dir.parent / "Datasets",
        Path("./Datasets"),
        Path("../Datasets")
    ]
    for cand in candidates:
        if cand.exists():
            csv_file = cand / "esc50.csv"
            audio_dir = cand / "audio"
            if csv_file.exists() and audio_dir.exists():
                return csv_file.resolve(), audio_dir.resolve()
    raise FileNotFoundError("Could not find ESC-50 dataset with esc50.csv and audio/ folder.")


def get_label_mapping(csv_path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    df = pd.read_csv(csv_path)
    unique_cats = sorted(df["category"].unique())
    id_to_class = {i: cat for i, cat in enumerate(unique_cats)}
    class_to_id = {cat: i for i, cat in enumerate(unique_cats)}
    return id_to_class, class_to_id


def save_label_mapping(id_to_class: Dict[int, str], output_path: str):
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_to_class.items()}, f, indent=2)


def extract_features_for_file(file_path: Path) -> np.ndarray:
    y, sr = load_audio(str(file_path), target_sr=TARGET_SAMPLE_RATE)
    mel = compute_mel_spectrogram(y, sr=sr, n_mels=128, target_frames=216)
    return np.expand_dims(mel, axis=-1).astype(np.float32)


def load_or_create_dataset_cache(
    force_recompute: bool = False,
    cache_path: Optional[Path] = None
) -> Dict[str, Any]:
    csv_path, audio_dir = find_dataset_paths()
    if cache_path is None:
        cache_path = exp_dir / "cache" / "esc50_features_cache_task509.npz"

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not force_recompute:
        try:
            print(f"[DatasetLoader] Loading cached features from: {cache_path}")
            cached = np.load(str(cache_path), allow_pickle=True)
            X_train = cached["X_train"]
            y_train = cached["y_train"]
            X_val = cached["X_val"]
            y_val = cached["y_val"]
            X_test = cached["X_test"]
            y_test = cached["y_test"]
            class_names = cached["class_names"].tolist()
            id_to_class = {i: c for i, c in enumerate(class_names)}
            class_to_id = {c: i for i, c in enumerate(class_names)}

            if len(X_train) == 1200 and len(X_val) == 400 and len(X_test) == 400:
                print(f"[DatasetLoader] Cache verified: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
                return {
                    "X_train": X_train, "y_train": y_train,
                    "X_val": X_val, "y_val": y_val,
                    "X_test": X_test, "y_test": y_test,
                    "class_names": class_names,
                    "id_to_class": id_to_class,
                    "class_to_id": class_to_id,
                    "csv_path": str(csv_path),
                    "audio_dir": str(audio_dir)
                }
        except Exception as e:
            print(f"[DatasetLoader] Cache load failed ({e}), recomputing...")

    print(f"[DatasetLoader] Extracting Task 509 Mel Spectrograms ([-1.0, +1.0] scale) from {csv_path}...")
    df = pd.read_csv(csv_path)
    id_to_class, class_to_id = get_label_mapping(csv_path)
    class_names = [id_to_class[i] for i in range(len(id_to_class))]

    train_rows = df[df["fold"].isin([1, 2, 3])].copy().reset_index(drop=True)
    val_rows = df[df["fold"] == 4].copy().reset_index(drop=True)
    test_rows = df[df["fold"] == 5].copy().reset_index(drop=True)

    def process_subset(sub_df: pd.DataFrame, subset_name: str) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        features = []
        labels = []
        filenames = []
        categories = []
        total = len(sub_df)

        for idx, row in enumerate(sub_df.itertuples()):
            fname = row.filename
            cat = row.category
            target_idx = class_to_id[cat]
            wav_path = audio_dir / fname

            if not wav_path.exists():
                raise FileNotFoundError(f"Missing audio file: {wav_path}")

            feat = extract_features_for_file(wav_path)
            features.append(feat)
            labels.append(target_idx)
            filenames.append(fname)
            categories.append(cat)

            if (idx + 1) % 200 == 0 or (idx + 1) == total:
                print(f"[DatasetLoader] {subset_name}: Processed {idx + 1}/{total} files...")

        return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32), filenames, categories

    print("[DatasetLoader] Processing Training Set (Folds 1, 2, 3)...")
    X_train, y_train, train_filenames, train_categories = process_subset(train_rows, "Train")

    print("[DatasetLoader] Processing Validation Set (Fold 4)...")
    X_val, y_val, val_filenames, val_categories = process_subset(val_rows, "Val")

    print("[DatasetLoader] Processing Test Set (Fold 5)...")
    X_test, y_test, test_filenames, test_categories = process_subset(test_rows, "Test")

    print(f"[DatasetLoader] Saving features to cache: {cache_path}")
    np.savez_compressed(
        str(cache_path),
        X_train=X_train, y_train=y_train, train_filenames=train_filenames, train_categories=train_categories,
        X_val=X_val, y_val=y_val, val_filenames=val_filenames, val_categories=val_categories,
        X_test=X_test, y_test=y_test, test_filenames=test_filenames, test_categories=test_categories,
        class_names=class_names
    )

    print(f"[DatasetLoader] Cache created successfully: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"[DatasetLoader] Value Range: Min={X_train.min():.4f}, Max={X_train.max():.4f}, Mean={X_train.mean():.4f}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "class_names": class_names,
        "id_to_class": id_to_class,
        "class_to_id": class_to_id,
        "csv_path": str(csv_path),
        "audio_dir": str(audio_dir)
    }

