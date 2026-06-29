import torch

from mnist.components.classifier import MNISTClassifier
from mnist.components.encoder import MNISTEncoder
from mnist.model import MNISTNet
from mnist.training.lightning_module import MNISTLightningModule
from research_core.lightning.lightning_trainer import StepOutputs
from research_core.schema.typed_tensor_dict import TypedTensorDict


def _make_module(label_smoothing: float = 0.0):
    encoder = MNISTEncoder(hidden_dim=32)
    classifier = MNISTClassifier(hidden_dim=32, num_classes=10)
    model = MNISTNet(encoder=encoder, classifier=classifier)
    config = {
        "learning_rate": 1e-3,
        "max_steps": 10,
        "label_smoothing": label_smoothing,
        "weight_decay": 0.0,
    }
    return MNISTLightningModule(model=model, trainer_config=config)


def test_training_step_returns_loss():
    module = _make_module()
    batch = TypedTensorDict(
        {
            "images": torch.randn(4, 1, 28, 28),
            "labels": torch.randint(0, 10, (4,)),
        },
        batch_size=[4],
    )
    out = module.training_step(batch, batch_idx=0)
    assert StepOutputs.LOSS in out
    assert out[StepOutputs.LOSS].requires_grad


def test_post_process_adds_predicted_classes():
    module = _make_module()
    predictions = {"logits": torch.randn(4, 10)}
    batch = {"images": torch.randn(4, 1, 28, 28), "labels": torch.randint(0, 10, (4,))}
    result = module.post_process_predictions(batch, predictions, batch_idx=0)
    assert "predicted_classes" in result
    assert result["predicted_classes"].shape == (4,)


def test_label_smoothing_configures_correct_loss():
    module = _make_module(label_smoothing=0.1)
    loss_fn = module.configure_loss()
    preds = TypedTensorDict({"logits": torch.randn(4, 10)}, batch_size=[4])
    batch = TypedTensorDict({"labels": torch.randint(0, 10, (4,))}, batch_size=[4])
    out = loss_fn(batch, preds)
    assert "label_smoothing_ce" in out


def test_no_label_smoothing_configures_ce_loss():
    module = _make_module(label_smoothing=0.0)
    loss_fn = module.configure_loss()
    preds = TypedTensorDict({"logits": torch.randn(4, 10)}, batch_size=[4])
    batch = TypedTensorDict({"labels": torch.randint(0, 10, (4,))}, batch_size=[4])
    out = loss_fn(batch, preds)
    assert "cross_entropy" in out
