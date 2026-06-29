import torch
from torch.utils.data import Dataset, random_split
from torchvision import datasets, transforms

from research_core.schema.data.schema_datamodule import SchemaDataModule
from research_core.schema.tensor_schema import TensorSchema
from research_core.schema.typed_tensor_dict import TypedTensorDict

from mnist.config import MNISTTrainingConfig
from mnist.schema.schemas import MNIST_PRODUCER_SCHEMA


class _MNISTDataset(Dataset):
    """Thin wrapper that yields TypedTensorDict samples from a torchvision dataset."""

    def __init__(self, base_dataset: Dataset, schema: TensorSchema) -> None:
        self._base = base_dataset
        self._schema = schema

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> TypedTensorDict:
        image, label = self._base[idx]
        return TypedTensorDict(
            {"images": image, "labels": torch.tensor(label, dtype=torch.long)},
            batch_size=[],
            tensor_schema=self._schema,
        )


class MNISTDataModule(SchemaDataModule):
    """Schema-driven DataModule for MNIST with 90/10 train/val split."""

    def __init__(self, config: MNISTTrainingConfig) -> None:
        self._data_dir = config.data_dir
        super().__init__(
            output_schema=MNIST_PRODUCER_SCHEMA,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

    def _build_producer_schema(self) -> TensorSchema:
        return MNIST_PRODUCER_SCHEMA

    def _build_datasets(self, stage: str | None) -> None:
        transform = transforms.ToTensor()
        full_dataset = datasets.MNIST(
            root=self._data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        train_size = int(0.9 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_split, val_split = random_split(full_dataset, [train_size, val_size])

        self._train_dataset = _MNISTDataset(train_split, MNIST_PRODUCER_SCHEMA)
        self._val_dataset = _MNISTDataset(val_split, MNIST_PRODUCER_SCHEMA)
