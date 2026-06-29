from mnist.schema.schemas import (
    MNIST_INPUT_SCHEMA,
    MNIST_LABEL_SCHEMA,
    MNIST_OUTPUT_SCHEMA,
    MNIST_PRODUCER_SCHEMA,
)


def test_mnist_input_schema_shape():
    assert MNIST_INPUT_SCHEMA.tensor_types[0].dims == [-1, 1, 28, 28]


def test_mnist_output_schema_shape():
    assert MNIST_OUTPUT_SCHEMA.tensor_types[0].dims == [-1, 10]


def test_mnist_label_schema_shape():
    assert MNIST_LABEL_SCHEMA.tensor_types[0].dims == [-1]


def test_mnist_producer_schema_is_unbatched():
    images_tt = MNIST_PRODUCER_SCHEMA.tensor_types[0]
    axis_type_names = [type(a).__name__ for a in images_tt.axes]
    assert "BatchAxis" not in axis_type_names


def test_mnist_producer_schema_has_labels():
    names = [tt.name for tt in MNIST_PRODUCER_SCHEMA.tensor_types]
    assert "images" in names
    assert "labels" in names
