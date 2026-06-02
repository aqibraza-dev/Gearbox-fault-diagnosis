import os
from torch.utils.tensorboard import SummaryWriter

class ExperimentTracker:
    """Unified tracker for MLflow and TensorBoard logging."""
    def __init__(self, config):
        self.config = config
        self.tb_writer = None
        self.use_mlflow = config.get('use_mlflow', False)
        self.use_tensorboard = config.get('use_tensorboard', False)

        # Initialize TensorBoard
        if self.use_tensorboard:
            log_dir = os.path.join(config.get('tensorboard_log_dir', 'logs'), config.get('run_name', 'run'))
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            print(f" TensorBoard initialized. Logging to: {log_dir}")

        # Initialize MLflow
        if self.use_mlflow:
            import mlflow
            mlflow.set_tracking_uri(config.get('mlflow_tracking_uri', 'mlruns'))
            mlflow.set_experiment(config.get('experiment_name', 'Default_Experiment'))
            print(f" MLflow initialized. Experiment: '{config['experiment_name']}'")

    def log_metrics(self, metrics, step, prefix=""):
        """Logs a dictionary of metrics at a specific training step or epoch."""
        prefixed_metrics = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        
        # Log to TensorBoard
        if self.use_tensorboard and self.tb_writer:
            for k, v in prefixed_metrics.items():
                self.tb_writer.add_scalar(k, v, global_step=step)

        # Log to MLflow
        if self.use_mlflow:
            import mlflow
            mlflow.log_metrics(prefixed_metrics, step=step)

    def log_params(self, params):
        """Logs static hyperparameters at the start of a run."""
        if self.use_mlflow:
            import mlflow
            # Filter out lists or complex objects for standard MLflow parameter compatibility
            clean_params = {k: str(v) for k, v in params.items() if not isinstance(v, (list, dict))}
            mlflow.log_params(clean_params)

    def close(self):
        """Flushes and closes tracking contexts."""
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.close()