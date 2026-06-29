import torch

from research_core.lightning.lightning_trainer import (
    CoreLightningModule,
    LossFn,
    StepOutputs,
)

from mnist.training.losses.cross_entropy_loss import CrossEntropyLoss
from mnist.training.losses.label_smoothing_loss import LabelSmoothingLoss


class MNISTLightningModule(CoreLightningModule):
    """Lightning module for MNIST training."""

    def configure_loss(self) -> LossFn:
        label_smoothing = self.trainer_config.get("label_smoothing", 0.0)
        if label_smoothing > 0:
            loss_obj = LabelSmoothingLoss(label_smoothing)
        else:
            loss_obj = CrossEntropyLoss()

        def loss_fn(batch, predictions):
            return loss_obj(batch=batch, predictions=predictions)

        return loss_fn

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.trainer_config["learning_rate"],
            weight_decay=self.trainer_config.get("weight_decay", 0.0),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer_config["max_steps"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def post_process_predictions(self, batch, predictions, batch_idx):
        predictions["predicted_classes"] = torch.argmax(predictions["logits"], dim=1)
        return predictions
