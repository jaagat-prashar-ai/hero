from mnist.config import MNISTTrainingConfig
from mnist.training.train import _flat_to_config


def test_flat_to_config_defaults():
    config = _flat_to_config({})
    assert config.batch_size == 64
    assert config.resume_run_id is None
    assert config.checkpoint_artifact is None
    assert config.label_smoothing == 0.0
    assert config.encoder.hidden_dim == 128
    assert config.encoder.dropout == 0.0


def test_flat_to_config_encoder_nested():
    config = _flat_to_config({"hidden_dim": 256, "dropout": 0.1})
    assert config.encoder.hidden_dim == 256
    assert config.encoder.dropout == 0.1


def test_flat_to_config_overrides():
    config = _flat_to_config({
        "batch_size": 128,
        "learning_rate": 3e-4,
        "label_smoothing": 0.1,
        "max_steps": 1000,
    })
    assert config.batch_size == 128
    assert config.learning_rate == 3e-4
    assert config.label_smoothing == 0.1
    assert config.max_steps == 1000


def test_flat_to_config_resume_fields():
    config = _flat_to_config({
        "resume_run_id": "abc123",
        "checkpoint_artifact": "research/mnist-template/model-abc:v3",
    })
    assert config.resume_run_id == "abc123"
    assert config.checkpoint_artifact == "research/mnist-template/model-abc:v3"


def test_flat_to_config_returns_training_config_type():
    config = _flat_to_config({})
    assert isinstance(config, MNISTTrainingConfig)
