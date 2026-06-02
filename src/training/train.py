import os
import torch
from src.utils.config import load_config
from src.utils.logger import ExperimentTracker
from src.data.dataset import load_and_prepare_data
from src.models.cnn import ConditionalGenerator, ConditionalCritic, FaultClassifier
from src.models.transformer import LSTMClassifier, AdaSTNetClassifier, FTClassifier
from src.training.trainer import train_wgangp_stage, train_classifier_stage
from src.evaluation.metrics import evaluate_on_test

def main():
    config = load_config()
    
    # Initialize the tracking core wrapper
    tracker = ExperimentTracker(config)
    
    # If MLflow is active, wrap execution context to maintain synchronization logs
    if config.get('use_mlflow', False):
        import mlflow
        mlflow.start_run(run_name=config.get('run_name', 'fault_diagnosis_run'))
        tracker.log_params(config)

    print(f"--- Launching Experiment Engine on Hardware: {config['device'].upper()} ---")
    
    train_loader, test_loader = load_and_prepare_data(
        config['csv_files'], config['class_labels'], config
    )

    # Initialize Generative Models
    generator = ConditionalGenerator(
        latent_dim=config['latent_dim'], num_classes=config['num_classes'],
        sequence_length=config['sequence_length'], embed_dim=config['embed_dim']
    ).to(config['device'])
    
    critic = ConditionalCritic(
        num_classes=config['num_classes'], sequence_length=config['sequence_length'], embed_dim=config['embed_dim']
    ).to(config['device'])

    # Stage 1: WGAN-GP Training
    train_wgangp_stage(generator, critic, train_loader, config, tracker=tracker)

    # Dictionary of variant baselines to benchmark
    models = {
        'Proposed (WGAN+CNN)': FaultClassifier(num_classes=config['num_classes']).to(config['device']),
        'LSTM Baseline': LSTMClassifier(num_classes=config['num_classes']).to(config['device']),
        'AdaSTNet Proxy': AdaSTNetClassifier(num_classes=config['num_classes']).to(config['device']),
        'FT-CNN Baseline': FTClassifier(num_classes=config['num_classes']).to(config['device'])
    }
    
    # Stage 2: Classifier Training & Benchmark Evaluation Loop
    os.makedirs(config['save_dir'], exist_ok=True)
    for name, model in models.items():
        augment_flag = True if 'Proposed' in name else False
        
        train_classifier_stage(
            generator, model, train_loader, config, 
            model_name=name, augment=augment_flag, tracker=tracker
        )
        
        # Check validation accuracy against hidden evaluation loader
        acc = evaluate_on_test(model, test_loader, config['device'])
        print(f">> Evaluation Completed -> {name} Test Accuracy: {acc:.4f}")
        
        # Track summary evaluations directly to current run log tracking metrics
        if config.get('use_mlflow', False) or config.get('use_tensorboard', False):
            tracker.log_metrics({f"test_accuracy": acc}, step=config['epochs_stage_2'], prefix=name.replace(" ", "_"))

        # Save model weights
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").lower()
        torch.save(model.state_dict(), os.path.join(config['save_dir'], f"{safe_name}.pth"))
        
    torch.save(generator.state_dict(), os.path.join(config['save_dir'], "wgan_generator.pth"))
    
    # Safely close open run resources
    tracker.close()
    if config.get('use_mlflow', False):
        mlflow.end_run()
        
    print("\nExperiment execution complete. Logs and track outputs saved correctly.")

if __name__ == "__main__":
    main()