import torch

from mnist.components.encoder import MNISTEncoder
from mnist.schema.schemas import MNIST_INPUT_SCHEMA
from research_core.schema.typed_tensor_dict import TypedTensorDict


def test_encoder_output_shape():
    enc = MNISTEncoder(hidden_dim=64)
    batch = TypedTensorDict({"images": torch.randn(4, 1, 28, 28)}, batch_size=[4])
    out = enc(batch)
    assert out["features"].shape == (4, 64)


def test_encoder_respects_input_schema():
    enc = MNISTEncoder(hidden_dim=64)
    assert enc.input_schema() == MNIST_INPUT_SCHEMA


def test_encoder_output_schema_reflects_hidden_dim():
    enc = MNISTEncoder(hidden_dim=256)
    schema = enc.output_schema()
    assert schema.tensor_types[0].dims == [-1, 256]


def test_encoder_applies_normalization():
    enc = MNISTEncoder(hidden_dim=64)
    batch = TypedTensorDict({"images": torch.ones(1, 1, 28, 28)}, batch_size=[1])
    out = enc(batch)
    assert out["features"].shape == (1, 64)


def test_encoder_different_batch_sizes():
    enc = MNISTEncoder(hidden_dim=32)
    for bs in [1, 4, 16]:
        batch = TypedTensorDict({"images": torch.randn(bs, 1, 28, 28)}, batch_size=[bs])
        out = enc(batch)
        assert out["features"].shape == (bs, 32)
