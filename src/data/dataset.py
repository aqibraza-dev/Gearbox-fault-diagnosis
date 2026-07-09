import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


def load_config(config_or_path=None):
    if config_or_path is None:
        config_path = DEFAULT_CONFIG_PATH
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    if isinstance(config_or_path, (str, Path)):
        config_path = Path(config_or_path)
        if not config_path.is_absolute():
            candidate = PROJECT_ROOT / config_path
            if candidate.exists():
                config_path = candidate
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    if isinstance(config_or_path, dict):
        return config_or_path

    raise TypeError("config must be a dict, a path, or None")


def _resolve_csv_path(csv_path):
    path = Path(csv_path)
    if path.exists():
        return path

    candidates = [
        PROJECT_ROOT / path,
        PROJECT_ROOT / "data" / path.name,
        PROJECT_ROOT / "data" / "raw" / path.name,
        PROJECT_ROOT / "dataset" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return path


def _window_signal(raw_data, sequence_length, stride):
    windows = []
    num_samples = raw_data.shape[0]

    for i in range(0, num_samples - sequence_length + 1, stride):
        window = raw_data[i : i + sequence_length].T
        windows.append(window)

    if not windows:
        return np.empty((0, raw_data.shape[1], sequence_length), dtype=np.float32)

    return np.asarray(windows, dtype=np.float32)


def _split_windows(windows, label, test_size=0.5, seed=42):
    if len(windows) == 0:
        return np.empty((0,)), np.empty((0,)), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(windows))

    if len(indices) == 1:
        train_idx = indices
        test_idx = np.array([], dtype=int)
    else:
        split_idx = int(len(indices) * (1.0 - test_size))
        split_idx = max(1, min(split_idx, len(indices) - 1))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

    x_train = windows[train_idx]
    x_test = windows[test_idx]
    y_train = np.full((len(x_train),), label, dtype=np.int64)
    y_test = np.full((len(x_test),), label, dtype=np.int64)

    return x_train, x_test, y_train, y_test


def _cache_split(csv_paths, labels_list, config, split_root):
    sequence_length = int(config.get("sequence_length", 1024))
    stride = int(config.get("stride", 512))
    test_size = float(config.get("test_size", 0.5))
    seed = int(config.get("seed", 42))

    train_dir = split_root / "train"
    test_dir = split_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for csv_path, label in zip(csv_paths, labels_list):
        resolved_path = _resolve_csv_path(csv_path)
        if not resolved_path.exists():
            print(f"Warning: Data file {resolved_path} not found. Skipping.")
            continue

        df = pd.read_csv(resolved_path)
        raw_data = df.iloc[:, 0:4].values

        windows = _window_signal(raw_data, sequence_length, stride)
        if len(windows) == 0:
            continue

        x_train, x_test, y_train, y_test = _split_windows(
            windows, label, test_size=test_size, seed=seed
        )

        stem = resolved_path.stem
        torch.save(
            {
                "x": torch.tensor(x_train, dtype=torch.float32),
                "y": torch.tensor(y_train, dtype=torch.long),
            },
            train_dir / f"{stem}_label_{label}.pt",
        )
        torch.save(
            {
                "x": torch.tensor(x_test, dtype=torch.float32),
                "y": torch.tensor(y_test, dtype=torch.long),
            },
            test_dir / f"{stem}_label_{label}.pt",
        )


def _load_split_loader(split_dir, batch_size, shuffle=False, drop_last=False):
    if not split_dir.exists():
        return None

    x_list, y_list = [], []
    for file_path in sorted(split_dir.glob("*.pt")):
        payload = torch.load(file_path, map_location="cpu")
        x_list.append(payload["x"])
        y_list.append(payload["y"])

    if not x_list:
        return None

    x_tensor = torch.cat(x_list, dim=0)
    y_tensor = torch.cat(y_list, dim=0)

    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def load_and_prepare_data(
    csv_paths=None,
    labels_list=None,
    config_or_path=None,
    test_only=False,
    signal_type="current",
):
    """
    Loads signal data, splits it into train/test folders, and returns DataLoaders.

    If csv_paths/labels_list are not provided, they are read from configs/train.yaml.
    """
    config = load_config(config_or_path)

    batch_size = int(config.get("batch_size", 32))
    split_root = Path(config.get("split_data_dir", PROJECT_ROOT / "data" / "split_dataset"))
    if not split_root.is_absolute():
        split_root = PROJECT_ROOT / split_root

    if csv_paths is None or labels_list is None:
        if signal_type == "vibration":
            csv_paths = config.get("csv_files_vibration", config.get("csv_files", []))
        else:
            csv_paths = config.get("csv_files_current", config.get("csv_files", []))
        labels_list = config.get("class_labels", list(range(len(csv_paths))))

    if not csv_paths:
        raise RuntimeError(
            "No CSV files configured. Set 'csv_files' in configs/train.yaml "
            "or pass csv_paths explicitly."
        )

    if not test_only:
        print("Preparing train/test split folders...")
        _cache_split(csv_paths, labels_list, config, split_root)

    train_dir = split_root / "train"
    test_dir = split_root / "test"

    if test_only:
        test_loader = _load_split_loader(
            test_dir,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        return test_loader

    train_loader = _load_split_loader(
        train_dir,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = _load_split_loader(
        test_dir,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    if train_loader is None or test_loader is None:
        raise RuntimeError("Split dataset folders are empty or missing.")

    print("Data ready from split folders.")
    return train_loader, test_loader