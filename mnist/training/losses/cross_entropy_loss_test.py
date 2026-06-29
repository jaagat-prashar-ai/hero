import torch

from mnist.training.losses.cross_entropy_loss import CrossEntropyLoss
from mnist.training.losses.label_smoothing_loss import LabelSmoothingLoss
from research_core.lightning.lightning_trainer import StepOutputs
from research_core.schema.typed_tensor_dict import TypedTensorDict


def test_cross_entropy_loss_scalar():
    loss_fn = CrossEntropyLoss()
    preds = TypedTensorDict({"logits": torch.randn(4, 10)}, batch_size=[4])
    batch = TypedTensorDict({"labels": torch.randint(0, 10, (4,))}, batch_size=[4])
    out = loss_fn(batch=batch, predictions=preds)
    assert out[StepOutputs.LOSS].shape == torch.Size([])
    assert "cross_entropy" in out


def test_label_smoothing_loss_scalar():
    loss_fn = LabelSmoothingLoss(smoothing=0.1)
    preds = TypedTensorDict({"logits": torch.randn(4, 10)}, batch_size=[4])
    batch = TypedTensorDict({"labels": torch.randint(0, 10, (4,))}, batch_size=[4])
    out = loss_fn(batch=batch, predictions=preds)
    assert out[StepOutputs.LOSS].shape == torch.Size([])
    assert "label_smoothing_ce" in out


def test_label_smoothing_loss_differs_from_ce():
    torch.manual_seed(42)
    preds = TypedTensorDict({"logits": torch.randn(4, 10)}, batch_size=[4])
    labels = TypedTensorDict({"labels": torch.randint(0, 10, (4,))}, batch_size=[4])
    ce = CrossEntropyLoss()
    ls = LabelSmoothingLoss(smoothing=0.1)
    ce_loss = ce(batch=labels, predictions=preds)[StepOutputs.LOSS]
    ls_loss = ls(batch=labels, predictions=preds)[StepOutputs.LOSS]
    assert ce_loss != ls_loss
