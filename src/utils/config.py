from pathlib import Path

import yaml
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "train.yaml"


def load_config(config_path=None):
    """Reads configuration properties seamlessly with fallbacks."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)
        if not config_path.is_absolute():
            candidate = PROJECT_ROOT / config_path
            if candidate.exists():
                config_path = candidate

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if config.get("device") == "cuda" and not torch.cuda.is_available():
        config["device"] = "cpu"

    return config