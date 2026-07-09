import os
import re
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import ExperimentTracker
from src.data.dataset import load_and_prepare_data
from src.models.cnn import ConditionalGenerator, ConditionalCritic, FaultClassifier
from src.models.transformer import LSTMClassifier, AdaSTNetClassifier, FTClassifier, FNO1DClassifier, FNet1DClassifier, PhaseMagnitudeNet, DualStreamSpectralNet, STFTSpectrogramCNN, SpectralAttentionNet, ComplexSpectralCNN, DilatedFourierCNN, SE_FourierNet, FourierResidualNet, WaveletSimCNN, GaborSpectralNet, STFT2DClassifier, DeepFNO1DClassifier64, DeepFNO1DClassifier32
from src.training.trainer import train_wgangp_stage, train_classifier_stage
from src.evaluation.metrics import evaluate_on_test


def _get_next_model_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)

    max_idx = 0
    pattern = re.compile(r"^model(\d+)$", re.IGNORECASE)

    for entry in base_dir.iterdir():
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))

    next_dir = base_dir / f"model{max_idx + 1:02d}"
    next_dir.mkdir(parents=True, exist_ok=False)
    return next_dir


def main():
    config = load_config()

    tracker = ExperimentTracker(config)

    if config.get("use_mlflow", False) and tracker.mlflow:
        tracker.mlflow.start_run(run_name=config.get("run_name", "fault_diagnosis_run"))
        tracker.log_params(config)

    print(f"--- Launching Experiment Engine on Hardware: {config['device'].upper()} ---")

    train_loader, test_loader = load_and_prepare_data(
        config["csv_files"], config["class_labels"], config
    )

    generator = ConditionalGenerator(
        latent_dim=config["latent_dim"],
        num_classes=config["num_classes"],
        sequence_length=config["sequence_length"],
        embed_dim=config["embed_dim"],
    ).to(config["device"])

    critic = ConditionalCritic(
        num_classes=config["num_classes"],
        sequence_length=config["sequence_length"],
        embed_dim=config["embed_dim"],
    ).to(config["device"])

    train_wgangp_stage(generator, critic, train_loader, config, tracker=tracker)

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
    save_dir = _get_next_model_dir(models_root)

    for name, model in models.items():
        augment_flag = "Proposed" in name

        train_classifier_stage(
            generator, model, train_loader, config,
            model_name=name, augment=augment_flag, tracker=tracker
        )

        acc = evaluate_on_test(model, test_loader, config["device"])
        print(f">> Evaluation Completed -> {name} Test Accuracy: {acc:.4f}")

        if config.get("use_mlflow", False) or config.get("use_tensorboard", False):
            tracker.log_metrics({"test_accuracy": acc}, step=config["epochs_stage_2"], prefix=name.replace(" ", "_"))

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").lower()
        torch.save(model.state_dict(), os.path.join(save_dir, f"{safe_name}.pth"))

    torch.save(generator.state_dict(), os.path.join(save_dir, "wgan_generator.pth"))

    tracker.close()
    if config.get("use_mlflow", False) and tracker.mlflow:
        tracker.mlflow.end_run()

    print(f"\nExperiment execution complete. Models saved in: {save_dir}")


if __name__ == "__main__":
    main()