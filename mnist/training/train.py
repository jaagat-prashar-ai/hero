"""MNIST training loop entrypoint for Lilypad."""

from typing import Any

import wandb
from lilypad.public.experiment_tracking.experiment_tracker_base import ExperimentTracker

from research_core.lightning.callbacks.image_logging.wandb_log_image import (
    SchemaWandbLogImageCallback,
)
from research_core.lightning.callbacks.losses.loss_logging import LossLoggingCallback
from research_core.lightning.callbacks.metrics.classification_metrics import (
    ClassificationMetricsCallback,
)
from research_core.lightning.lightning_trainer import CoreLightningTrainer
from research_core.lightning.preemption import fetch_latest_checkpoint, resume_wandb_run

from mnist.components.classifier import MNISTClassifier
from mnist.components.encoder import MNISTEncoder
from mnist.config import EncoderConfig, MNISTTrainingConfig
from mnist.data.datamodule import MNISTDataModule
from mnist.model import MNISTNet
from mnist.training.callbacks.confusion_matrix_callback import ConfusionMatrixCallback
from mnist.training.lightning_module import MNISTLightningModule


def _flat_to_config(flat: dict[str, Any]) -> MNISTTrainingConfig:
    """Convert flat Lilypad YAML dict to nested MNISTTrainingConfig."""
    return MNISTTrainingConfig(
        encoder=EncoderConfig(
            hidden_dim=flat.get("hidden_dim", 128),
            dropout=flat.get("dropout", 0.0),
        ),
        num_classes=flat.get("num_classes", 10),
        label_smoothing=flat.get("label_smoothing", 0.0),
        batch_size=flat.get("batch_size", 64),
        num_workers=flat.get("num_workers", 4),
        data_dir=flat.get("data_dir", "/tmp/mnist_data"),
        learning_rate=flat.get("learning_rate", 1e-4),
        weight_decay=flat.get("weight_decay", 0.0),
        max_steps=flat.get("max_steps", 5000),
        val_check_interval=flat.get("val_check_interval", 200),
        checkpoint_interval=flat.get("checkpoint_interval", 500),
        log_interval=flat.get("log_interval", 20),
        image_log_interval=flat.get("image_log_interval", 100),
        accelerator=flat.get("accelerator", "gpu"),
        devices=int(flat.get("devices", 1)),
        num_nodes=int(flat.get("num_nodes", 1)),
        strategy=flat.get("strategy", "auto"),
        precision=flat.get("precision", "32-true"),
        wandb_project=flat.get("wandb_project", "mnist-template"),
        wandb_entity=flat.get("wandb_entity", "research"),
        resume_run_id=flat.get("resume_run_id"),
        checkpoint_artifact=flat.get("checkpoint_artifact"),
    )


def training_loop(
    training_fn_config: dict[str, Any], experiment_tracker: ExperimentTracker
) -> None:
    """Lilypad-compatible training loop entrypoint.

    WandB is initialized by Lilypad before this function is called.

    Preemption resumption:
    - When requeue_if_preempted=true, Lilypad restarts with the same W&B run ID,
      so fetch_latest_checkpoint finds the prior checkpoint automatically.
    - Set resume_run_id only as a manual fallback for explicit re-launches.
    """
    config = _flat_to_config(training_fn_config)

    # Preemption resumption from research_core — not duplicated here.
    ckpt_path = fetch_latest_checkpoint(wandb.run.id, wandb.run.entity, wandb.run.project)
    if ckpt_path is None and config.resume_run_id is not None:
        ckpt_path = resume_wandb_run(
            config.resume_run_id, config.wandb_entity, config.wandb_project
        )

    encoder = MNISTEncoder(
        hidden_dim=config.encoder.hidden_dim,
        dropout=config.encoder.dropout,
    )
    classifier = MNISTClassifier(
        hidden_dim=config.encoder.hidden_dim,
        num_classes=config.num_classes,
    )
    model = MNISTNet(encoder=encoder, classifier=classifier)
    datamodule = MNISTDataModule(config=config)

    module = MNISTLightningModule.create_or_load(
        model=model,
        trainer_config=vars(config),
        checkpoint_artifact=config.checkpoint_artifact,
    )

    callbacks = [
        LossLoggingCallback(every_n_steps=config.log_interval),
        ClassificationMetricsCallback(
            num_classes=config.num_classes,
            prediction_key="predicted_classes",
            target_key="labels",
        ),
        ConfusionMatrixCallback(num_classes=config.num_classes),
        SchemaWandbLogImageCallback(
            batch_keys_to_log={"input_viz": "images"},
            prediction_keys_to_log={},
            every_n_steps=config.image_log_interval,
        ),
    ]

    trainer = CoreLightningTrainer(
        use_wandb=True,
        checkpoint_interval=config.checkpoint_interval,
        max_steps=config.max_steps,
        val_check_interval=config.val_check_interval,
        log_every_n_steps=config.log_interval,
        callbacks=callbacks,
        accelerator=config.accelerator,
        devices=config.devices,
        num_nodes=config.num_nodes,
        strategy=config.strategy,
        precision=config.precision,
        enable_progress_bar=False,
    )

    trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)
