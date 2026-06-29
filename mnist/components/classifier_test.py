import torch

from mnist.components.classifier import MNISTClassifier
from research_core.schema.typed_tensor_dict import TypedTensorDict


def test_classifier_output_shape():
    clf = MNISTClassifier(hidden_dim=64, num_classes=10)
    features = TypedTensorDict({"features": torch.randn(4, 64)}, batch_size=[4])
    out = clf(features)
    assert out["logits"].shape == (4, 10)


def test_classifier_custom_num_classes():
    clf = MNISTClassifier(hidden_dim=32, num_classes=5)
    features = TypedTensorDict({"features": torch.randn(2, 32)}, batch_size=[2])
    out = clf(features)
    assert out["logits"].shape == (2, 5)
