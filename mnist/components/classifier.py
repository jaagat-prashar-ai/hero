import torch.nn as nn

from research_core.proto.typed_tensors_pb2 import DataType
from research_core.schema.schema_module import SchemaModule
from research_core.schema.tensor_schema import Axis, TensorSchema, TensorType
from research_core.schema.typed_tensor_dict import TypedTensorDict

from mnist.schema.constants import FEATURE_DIM, NUM_CLASSES
from mnist.schema.schemas import MNIST_OUTPUT_SCHEMA


class MNISTClassifier(SchemaModule):
    """Linear classifier: [B, hidden_dim] -> [B, num_classes] logits."""

    def __init__(self, hidden_dim: int = FEATURE_DIM, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes
        self.fc = nn.Linear(hidden_dim, num_classes)

    def input_schema(self) -> TensorSchema:
        return TensorSchema(
            name="mnist_classifier_input",
            tensor_types=[
                TensorType(
                    name="features",
                    axes=[Axis.BatchAxis(), Axis.FeatureAxis()],
                    dims=[-1, self._hidden_dim],
                    dtype=DataType.FLOAT32,
                ),
            ],
        )

    @classmethod
    def output_schema(cls) -> TensorSchema:
        return MNIST_OUTPUT_SCHEMA

    def _forward_impl(self, inputs: TypedTensorDict) -> dict:
        logits = self.fc(inputs["features"])
        return {"logits": logits}
