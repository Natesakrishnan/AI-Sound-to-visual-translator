"""
Dataset Loader & Caching Engine for ESC-50 Environmental Sound Classification.
Strictly implements the official fold partitioning:
  - Folds 1, 2, 3 -> Training Set   (1,200 audio files)
  - Fold 4       -> Validation Set (  400 audio files)
  - Fold 5       -> Final Test Set (  400 audio files)
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict, List, Any, Optional, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure src directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_preprocessing import load_audio, TARGET_SAMPLE_RATE
from feature_extraction import compute_mel_spectrogram


# Verified 50 ESC-50 classes in alphabetical order
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

RANDOM_SEED = 42


def find_dataset_paths(custom_dir: Optional[str] = None) -> Tuple[Path, Path]:
    """
    Locate the ESC-50 dataset directory containing esc50.csv and audio/ folder.
    Searches multiple candidate locations automatically.
    """
    candidates = []
    if custom_dir:
        candidates.append(Path(custom_dir))

    # Current working dir and project root relative searches
    base_dir = Path(__file__).resolve().parent.parent
    candidates.extend([
        Path("K:/AI project/Datasets"),
        base_dir / "dataset",
        base_dir / "Datasets",
        base_dir.parent / "Datasets",
        Path("./dataset"),
        Path("./Datasets"),
        Path("../Datasets")
    ])

    for cand in candidates:
        if cand.exists():
            csv_file = cand / "esc50.csv"
            audio_dir = cand / "audio"
            if csv_file.exists() and audio_dir.exists():
                return csv_file.resolve(), audio_dir.resolve()

    raise FileNotFoundError(
        "Could not automatically locate ESC-50 dataset folder with esc50.csv and audio/.\n"
        f"Checked candidates: {[str(c) for c in candidates]}"
    )


def get_label_mapping(csv_path: Path) -> Tuple[Dict[int, str], Dict[str, int]]:
    """
    Build index-to-category and category-to-index mappings from the dataset CSV.
    """
    df = pd.read_csv(csv_path)
    unique_cats = sorted(df["category"].unique())
    id_to_class = {i: cat for i, cat in enumerate(unique_cats)}
    class_to_id = {cat: i for i, cat in enumerate(unique_cats)}
    return id_to_class, class_to_id


def save_label_mapping(id_to_class: Dict[int, str], output_path: str):
    """Save label mapping dictionary to JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in id_to_class.items()}, f, indent=2)


def load_label_mapping(json_path: str) -> Dict[int, str]:
    """Load label mapping from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def extract_features_for_file(file_path: Path) -> np.ndarray:
    """
    Extract standardized (128, 216, 1) Log-Mel Spectrogram for a single audio file.
    """
    y, sr = load_audio(str(file_path), target_sr=TARGET_SAMPLE_RATE)
    mel = compute_mel_spectrogram(y, sr=sr, n_mels=128, target_frames=216)
    # Add channel dimension
    mel_tensor = np.expand_dims(mel, axis=-1).astype(np.float32)
    return mel_tensor


def load_or_create_dataset_cache(
    dataset_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    force_recompute: bool = False,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> Dict[str, Any]:
    """
    Loads precomputed Mel Spectrogram dataset from cache if available;
    otherwise extracts features from all 2,000 files, caches to disk, and returns the dataset splits.

    Dataset Partitioning:
      - Train: Folds 1, 2, 3 (1,200 files)
      - Val:   Fold 4        (  400 files)
      - Test:  Fold 5        (  400 files)
    """
    csv_path, audio_dir = find_dataset_paths(dataset_dir)
    project_root = Path(__file__).resolve().parent.parent

    if cache_dir is None:
        cache_path = project_root / "outputs" / "cache" / "esc50_features_cache.npz"
    else:
        cache_path = Path(cache_dir) / "esc50_features_cache.npz"

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if valid cache exists
    if cache_path.exists() and not force_recompute:
        try:
            print(f"[DatasetLoader] Loading precomputed cached features from: {cache_path}")
            cached = np.load(str(cache_path), allow_pickle=True)
            X_train = cached["X_train"]
            y_train = cached["y_train"]
            X_val = cached["X_val"]
            y_val = cached["y_val"]
            X_test = cached["X_test"]
            y_test = cached["y_test"]
            test_filenames = cached["test_filenames"].tolist()
            test_categories = cached["test_categories"].tolist()
            class_names = cached["class_names"].tolist()
            id_to_class = {i: c for i, c in enumerate(class_names)}
            class_to_id = {c: i for i, c in enumerate(class_names)}

            # Validate cache sanity
            if len(X_train) == 1200 and len(X_val) == 400 and len(X_test) == 400:
                print(f"[DatasetLoader] Cache successfully verified: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
                return {
                    "X_train": X_train, "y_train": y_train,
                    "X_val": X_val, "y_val": y_val,
                    "X_test": X_test, "y_test": y_test,
                    "test_filenames": test_filenames,
                    "test_categories": test_categories,
                    "class_names": class_names,
                    "id_to_class": id_to_class,
                    "class_to_id": class_to_id,
                    "csv_path": str(csv_path),
                    "audio_dir": str(audio_dir)
                }
        except Exception as e:
            print(f"[DatasetLoader] Cache corrupted or outdated ({e}). Recomputing from scratch...")

    # Compute features from raw audio files
    print(f"[DatasetLoader] Extracting Mel Spectrograms from {csv_path}...")
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
        print(f"[DatasetLoader] Extracting {subset_name} ({total} files)...", flush=True)

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
                print(f"[DatasetLoader]   -> {subset_name}: {idx + 1}/{total} extracted", flush=True)

            if progress_callback:
                progress_callback(idx + 1, total, f"Extracting {subset_name} ({idx+1}/{total})")

        return (
            np.array(features, dtype=np.float32),
            np.array(labels, dtype=np.int32),
            filenames,
            categories
        )

    X_train, y_train, train_fn, train_cat = process_subset(train_rows, "Train Folds 1-3")
    X_val, y_val, val_fn, val_cat = process_subset(val_rows, "Val Fold 4")
    X_test, y_test, test_filenames, test_categories = process_subset(test_rows, "Test Fold 5")

    # Save to compressed cache
    print(f"[DatasetLoader] Saving features to cache: {cache_path}")
    np.savez_compressed(
        str(cache_path),
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        test_filenames=np.array(test_filenames),
        test_categories=np.array(test_categories),
        class_names=np.array(class_names)
    )

    # Also save label mapping to models directory
    models_dir = project_root / "models"
    save_label_mapping(id_to_class, str(models_dir / "label_mapping.json"))

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "test_filenames": test_filenames,
        "test_categories": test_categories,
        "class_names": class_names,
        "id_to_class": id_to_class,
        "class_to_id": class_to_id,
        "csv_path": str(csv_path),
        "audio_dir": str(audio_dir)
    }


if __name__ == "__main__":
    data = load_or_create_dataset_cache()
    print("X_train shape:", data["X_train"].shape, "y_train shape:", data["y_train"].shape)
    print("X_val shape:", data["X_val"].shape, "y_val shape:", data["y_val"].shape)
    print("X_test shape:", data["X_test"].shape, "y_test shape:", data["y_test"].shape)
    print("Total classes:", len(data["class_names"]))
