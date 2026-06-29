import lightning as pl
import torch
import wandb
from torchmetrics import ConfusionMatrix


class ConfusionMatrixCallback(pl.Callback):
    """Log a confusion matrix image to WandB at the end of each validation epoch."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self._num_classes = num_classes
        self._confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self._preds: list[torch.Tensor] = []
        self._targets: list[torch.Tensor] = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if outputs is None:
            return
        predictions = outputs.get("predictions", {})
        if "predicted_classes" not in predictions or "labels" not in batch:
            return
        self._preds.append(predictions["predicted_classes"].detach().cpu())
        self._targets.append(batch["labels"].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._preds:
            return
        all_preds = torch.cat(self._preds)
        all_targets = torch.cat(self._targets)
        cm = self._confusion_matrix(all_preds, all_targets)
        if wandb.run is not None:
            import matplotlib.pyplot as plt
            fig = self._make_figure(cm.numpy())
            wandb.log({"val/confusion_matrix": wandb.Image(fig)}, step=trainer.global_step)
            plt.close(fig)
        self._preds.clear()
        self._targets.clear()

    def _make_figure(self, cm):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        return fig
