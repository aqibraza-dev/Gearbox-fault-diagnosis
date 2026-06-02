import os
import yaml
import torch

def load_config(config_path="configs/train.yaml"):
    """Reads configuration properties seamlessly with fallbacks."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Resolving runtime hardware parameters
    if config.get('device') == 'cuda' and not torch.cuda.is_available():
        config['device'] = 'cpu'
        
    return config