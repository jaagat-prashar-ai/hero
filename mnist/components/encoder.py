import torch
import torch.nn as nn
import torch.nn.functional as F

from research_core.proto.typed_tensors_pb2 import DataType
from research_core.schema.schema_module import SchemaModule
from research_core.schema.tensor_schema import Axis, TensorSchema, TensorType
from research_core.schema.typed_tensor_dict import TypedTensorDict

from mnist.schema.constants import MNIST_MEAN, MNIST_STD
from mnist.schema.schemas import MNIST_INPUT_SCHEMA


class MNISTEncoder(SchemaModule):
    """Convolutional encoder: [B, 1, 28, 28] -> [B, hidden_dim].

    Applies MNIST normalization internally so the schema boundary
    always carries images in [0, 1] float range.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.0) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(64 * 5 * 5, hidden_dim)

    @classmethod
    def input_schema(cls) -> TensorSchema:
        return MNIST_INPUT_SCHEMA

    def output_schema(self) -> TensorSchema:
        return TensorSchema(
            name="mnist_encoder_output",
            tensor_types=[
                TensorType(
                    name="features",
                    axes=[Axis.BatchAxis(), Axis.FeatureAxis()],
                    dims=[-1, self._hidden_dim],
                    dtype=DataType.FLOAT32,
                ),
            ],
        )

    def _forward_impl(self, inputs: TypedTensorDict) -> dict:
        x = (inputs["images"] - MNIST_MEAN) / MNIST_STD
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return {"features": x}
