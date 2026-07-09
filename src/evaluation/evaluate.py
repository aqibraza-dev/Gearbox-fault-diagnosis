import argparse
import os
import re
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.utils.config import load_config
from src.data.dataset import load_and_prepare_data
from src.models.cnn import FaultClassifier
from src.models.transformer import LSTMClassifier, AdaSTNetClassifier, FTClassifier, FNO1DClassifier, FNet1DClassifier, PhaseMagnitudeNet, DualStreamSpectralNet, STFTSpectrogramCNN, SpectralAttentionNet, ComplexSpectralCNN, DilatedFourierCNN, SE_FourierNet, FourierResidualNet, WaveletSimCNN, GaborSpectralNet, STFT2DClassifier, DeepFNO1DClassifier64, DeepFNO1DClassifier32
from src.evaluation.visualization import evaluate_and_plot_all


def _safe_name(name: str):
    return (
        name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
        .lower()
    )


def _candidate_filenames(name: str):
    base = _safe_name(name)
    return [
        f"{base}.pth",
        f"{base.replace('plus', '')}.pth",
        f"{base.replace('plus', '_')}.pth",
    ]


def _get_latest_model_dir(models_root: Path) -> Path:
    if not models_root.exists():
        raise FileNotFoundError(f"Models root directory not found: {models_root}")

    pattern = re.compile(r"^model(\d+)$", re.IGNORECASE)
    candidates = []

    for entry in models_root.iterdir():
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                candidates.append((int(match.group(1)), entry))

    if not candidates:
        raise FileNotFoundError(f"No model subfolders found in: {models_root}")

    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _resolve_model_dir(models_root: Path, model_dir_arg: str | None) -> Path:
    if model_dir_arg:
        model_dir = Path(model_dir_arg)
        if not model_dir.is_absolute():
            model_dir = models_root / model_dir_arg
    else:
        model_dir = _get_latest_model_dir(models_root)

    if not model_dir.exists() or not model_dir.is_dir():
        raise FileNotFoundError(f"Model folder not found: {model_dir}")

    return model_dir


def _load_models_from_dir(models, save_dir, device):
    loaded_models = {}
    for name, model in models.items():
        found = None
        for fname in _candidate_filenames(name):
            path = os.path.join(save_dir, fname)
            if os.path.exists(path):
                found = path
                break

        if found:
            model.load_state_dict(torch.load(found, map_location=device, weights_only=True))
            print(f"Weights synced successfully for model: {name} -> {found}")
            loaded_models[name] = model
        else:
            print(f"CRITICAL WARNING: No checkpoint found for model: {name} in {save_dir}. Skipping.")
    return loaded_models


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models from models/modelXX folder.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Model subfolder under models/ (example: model01). If omitted, latest modelXX is used.",
    )
    args = parser.parse_args()

    config = load_config()
    test_loader = load_and_prepare_data(
        config["csv_files"], config["class_labels"], config, test_only=True
    )

    models = {
        "Proposed STFT-2D": STFT2DClassifier(num_classes=config["num_classes"]).to(config["device"]),
        "(WGAN+CNN)": FaultClassifier(num_classes=config["num_classes"]).to(config["device"]),
        "LSTM Baseline": LSTMClassifier(num_classes=config["num_classes"]).to(config["device"]),
        "AdaSTNet Proxy": AdaSTNetClassifier(num_classes=config["num_classes"]).to(config["device"]),
        "FT-CNN Baseline": FTClassifier(num_classes=config["num_classes"]).to(config["device"]),
        "FNO-1D Baseline": FNO1DClassifier(num_classes=config["num_classes"]).to(config["device"]),
        # "DeepFNO1DClassifier64": DeepFNO1DClassifier64(num_classes=config["num_classes"]).to(config["device"]),
        # "DeepFNO1DClassifier32": DeepFNO1DClassifier32(num_classes=config["num_classes"]).to(config["device"]),
        # "FNet-1D Baseline": FNet1DClassifier(num_classes=config["num_classes"]).to(config["device"]),
        # "Phase-Magnitude Net": PhaseMagnitudeNet(num_classes=config["num_classes"]).to(config["device"]),
        # "Dual-Stream Spectral Net": DualStreamSpectralNet(num_classes=config["num_classes"]).to(config["device"]),
        # "STFT Spectrogram CNN": STFTSpectrogramCNN(num_classes=config["num_classes"]).to(config["device"]),
        # "Spectral Attention Net": SpectralAttentionNet(num_classes=config["num_classes"]).to(config["device"]),
        # "Complex Spectral CNN": ComplexSpectralCNN(num_classes=config["num_classes"]).to(config["device"]),
        "Dilated Fourier CNN": DilatedFourierCNN(num_classes=config["num_classes"]).to(config["device"]),
        # "SE-FourierNet": SE_FourierNet(num_classes=config["num_classes"]).to(config["device"]),
        # "Fourier Residual Net": FourierResidualNet(num_classes=config["num_classes"]).to(config["device"]),
        "WaveletSimCNN": WaveletSimCNN(num_classes=config["num_classes"]).to(config["device"]),
        # "Gabor Spectral Net": GaborSpectralNet(num_classes=config["num_classes"]).to(config["device"]),
    }

    models_root = PROJECT_ROOT / "models"
    save_dir = _resolve_model_dir(models_root, args.model_dir)

    loaded_models = _load_models_from_dir(models, str(save_dir), config["device"])

    if not loaded_models:
        raise RuntimeError(f"No model checkpoints found in {save_dir}. Train models first.")

    run_name = config.get("run_name", "run").replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "results" / "images" / f"{timestamp}_{run_name}"

    evaluate_and_plot_all(
        loaded_models,
        test_loader,
        config["device"],
        config["num_classes"],
        output_dir=str(output_dir),
    )


if __name__ == "__main__":
    main()