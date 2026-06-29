from research_core.proto.typed_tensors_pb2 import DataType, TensorSemantic

from research_core.schema.tensor_schema import Axis, TensorSchema, TensorType

from mnist.schema.constants import IMAGE_H, IMAGE_W, NUM_CLASSES

# Per-sample (no batch dim) — used by MNISTDataModule
MNIST_PRODUCER_SCHEMA = TensorSchema(
    name="mnist_producer",
    tensor_types=[
        TensorType(
            name="images",
            axes=[Axis.ChannelAxis(), Axis.HeightAxis(), Axis.WidthAxis()],
            dims=[1, IMAGE_H, IMAGE_W],
            dtype=DataType.FLOAT32,
            semantic=TensorSemantic.IMAGE,
        ),
        TensorType(
            name="labels",
            axes=[],
            dims=[],
            dtype=DataType.INT64,
        ),
    ],
)

# Batched input schema — used by MNISTEncoder
MNIST_INPUT_SCHEMA = TensorSchema(
    name="mnist_input",
    tensor_types=[
        TensorType(
            name="images",
            axes=[Axis.BatchAxis(), Axis.ChannelAxis(), Axis.HeightAxis(), Axis.WidthAxis()],
            dims=[-1, 1, IMAGE_H, IMAGE_W],
            dtype=DataType.FLOAT32,
            semantic=TensorSemantic.IMAGE,
        ),
    ],
)

# Model output schema — logits [B, NUM_CLASSES]
MNIST_OUTPUT_SCHEMA = TensorSchema(
    name="mnist_output",
    tensor_types=[
        TensorType(
            name="logits",
            axes=[Axis.BatchAxis(), Axis.ClassAxis()],
            dims=[-1, NUM_CLASSES],
            dtype=DataType.FLOAT32,
        ),
    ],
)

# Label schema — used by losses
MNIST_LABEL_SCHEMA = TensorSchema(
    name="mnist_labels",
    tensor_types=[
        TensorType(
            name="labels",
            axes=[Axis.BatchAxis()],
            dims=[-1],
            dtype=DataType.INT64,
        ),
    ],
)
