import os
import re
from torch.utils.tensorboard import SummaryWriter

class ExperimentTracker:
    """Unified tracker for MLflow and TensorBoard logging."""
    def __init__(self, config):
        self.config = config
        self.tb_writer = None
        self.mlflow = None
        self.use_mlflow = config.get('use_mlflow', False)
        self.use_tensorboard = config.get('use_tensorboard', False)

        if self.use_tensorboard:
            log_dir = os.path.join(
                config.get('tensorboard_log_dir', 'logs'),
                config.get('run_name', 'run'),
            )
            self.tb_writer = SummaryWriter(log_dir=log_dir)
            print(f"TensorBoard initialized. Logging to: {log_dir}")

        if self.use_mlflow:
            try:
                import mlflow
                mlflow.set_tracking_uri(config.get('mlflow_tracking_uri', 'sqlite:///mlflow.db'))
                mlflow.set_experiment(config.get('experiment_name', 'Default_Experiment'))
                self.mlflow = mlflow
                print(f"MLflow initialized. Experiment: '{config.get('experiment_name', 'Default_Experiment')}'")
            except Exception as e:
                print(f"Warning: MLflow disabled: {e}")
                self.use_mlflow = False
                self.mlflow = None

    def _sanitize_key(self, key: str) -> str:
        # Keep only MLflow-allowed chars: alnum, _, -, ., space, :, /
        return re.sub(r"[^A-Za-z0-9_\-\. :/]", "_", key)

    def log_metrics(self, metrics, step, prefix=""):
        raw_metrics = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        prefixed_metrics = {self._sanitize_key(k): v for k, v in raw_metrics.items()}

        if self.use_tensorboard and self.tb_writer:
            for k, v in prefixed_metrics.items():
                self.tb_writer.add_scalar(k, v, global_step=step)

        if self.use_mlflow and self.mlflow:
            self.mlflow.log_metrics(prefixed_metrics, step=step)

    def log_params(self, params):
        if self.use_mlflow and self.mlflow:
            clean_params = {k: str(v) for k, v in params.items() if not isinstance(v, (list, dict))}
            self.mlflow.log_params(clean_params)

    def close(self):
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.close()