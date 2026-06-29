from research_core.schema.schema_module import SchemaModule
from research_core.schema.tensor_schema import TensorSchema
from research_core.schema.typed_tensor_dict import TypedTensorDict

from mnist.components.classifier import MNISTClassifier
from mnist.components.encoder import MNISTEncoder
from mnist.schema.schemas import MNIST_INPUT_SCHEMA, MNIST_OUTPUT_SCHEMA


class MNISTNet(SchemaModule):
    """Full MNIST model: encoder + classifier.

    Orchestrates MNISTEncoder and MNISTClassifier as independent
    SchemaModules, making each individually testable.
    """

    def __init__(self, encoder: MNISTEncoder, classifier: MNISTClassifier) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    @classmethod
    def input_schema(cls) -> TensorSchema:
        return MNIST_INPUT_SCHEMA

    @classmethod
    def output_schema(cls) -> TensorSchema:
        return MNIST_OUTPUT_SCHEMA

    def _forward_impl(self, inputs: TypedTensorDict) -> dict:
        features = self.encoder(inputs)
        logits = self.classifier(features)
        return {"logits": logits["logits"]}
