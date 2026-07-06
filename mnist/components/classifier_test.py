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


# use 8 workers for now to address the HF rate limit 
# bash launch file
# pre-launch validations, incase there are env variables missing 
# maintain a good launch file 
# 8 workers at most, 1 node, 8gpu node, think of it is a local machine, common temp folder
    # 8 workers...
# on rank 0, only download the files to /tmp

# exponentional backoff 
# need a good way of saving 

# maintain a separate folder
# for lilypad experiments 
# root_directory: . 
# zips hero folder, sends it as zip 

# modify it to a folder for wiring to work. 

# For lilypad, export env 

# create a ~/.credits/lilypad folder in home for main env variables
# in launch script, source from this location, never read the credentials