import torch.nn.functional as F

from research_core.lightning.lightning_trainer import StepOutputs
from research_core.schema.schema_loss import SchemaLoss
from research_core.schema.tensor_schema import TensorSchema
from research_core.schema.typed_tensor_dict import TypedTensorDict

from mnist.schema.schemas import MNIST_LABEL_SCHEMA, MNIST_OUTPUT_SCHEMA


class CrossEntropyLoss(SchemaLoss):
    """Standard cross-entropy loss over MNIST logits."""

    @classmethod
    def prediction_schema(cls) -> TensorSchema:
        return MNIST_OUTPUT_SCHEMA

    @classmethod
    def input_schema(cls) -> TensorSchema:
        return MNIST_LABEL_SCHEMA

    def _forward_impl(self, batch: TypedTensorDict, predictions: TypedTensorDict) -> dict:
        loss = F.cross_entropy(predictions["logits"], batch["labels"])
        return {StepOutputs.LOSS: loss, "cross_entropy": loss}
