import os
import torch
from src.utils.config import load_config
from src.data.dataset import load_and_prepare_data
from src.models.cnn import FaultClassifier
from src.models.transformer import LSTMClassifier, AdaSTNetClassifier, FTClassifier
from src.evaluation.visualization import evaluate_and_plot_all

def main():
    config = load_config()
    test_loader = load_and_prepare_data(
        config['csv_files'], config['class_labels'], config, test_only=True
    )

    models = {
        'Proposed (WGAN+CNN)': FaultClassifier(num_classes=config['num_classes']).to(config['device']),
        'LSTM Baseline': LSTMClassifier(num_classes=config['num_classes']).to(config['device']),
        'AdaSTNet Proxy': AdaSTNetClassifier(num_classes=config['num_classes']).to(config['device']),
        'FT-CNN Baseline': FTClassifier(num_classes=config['num_classes']).to(config['device'])
    }

    # Load trained weights back from the output checkpoint directory
    for name, model in models.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").lower()
        filepath = os.path.join(config['save_dir'], f"{safe_name}.pth")
        
        if os.path.exists(filepath):
            model.load_state_dict(torch.load(filepath, map_location=config['device'], weights_only=True))
            print(f"Weights synced successfully for model: {name}")
        else:
            print(f"CRITICAL WARNING: Checkpoint {filepath} missing. Model using uninitialized weights!")

    evaluate_and_plot_all(models, test_loader, config['device'], config['num_classes'])

if __name__ == "__main__":
    main()