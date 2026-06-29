from __future__ import annotations

from dataclasses import dataclass, field

from mnist.schema.constants import FEATURE_DIM, NUM_CLASSES


@dataclass
class EncoderConfig:
    hidden_dim: int = FEATURE_DIM
    dropout: float = 0.0


@dataclass
class MNISTTrainingConfig:
    # Model
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    num_classes: int = NUM_CLASSES

    # Loss
    label_smoothing: float = 0.0

    # Data
    batch_size: int = 64
    num_workers: int = 4
    data_dir: str = "/tmp/mnist_data"

    # Optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.0

    # Training schedule
    max_steps: int = 5000
    val_check_interval: int = 200
    checkpoint_interval: int = 500
    log_interval: int = 20
    image_log_interval: int = 100

    # Runtime
    accelerator: str = "gpu"
    devices: int = 1
    num_nodes: int = 1
    strategy: str = "auto"
    precision: str = "32-true"

    # Logging
    wandb_project: str = "mnist-template"
    wandb_entity: str = "research"

    # Resumption
    resume_run_id: str | None = None
    checkpoint_artifact: str | None = None
