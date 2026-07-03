"""Shared stubs and helpers for the ``pyvisim.neural_networks`` test suite."""

from __future__ import annotations

import os
from unittest import mock

import numpy as np
import torch

from pyvisim.neural_networks import SiameseNeuralNetwork
from pyvisim.typing import UInt8NumpyArray


class FlattenBackbone(torch.nn.Module):
    """Backbone stub that flattens its input, leaving values untouched.

    Feeding a ``(batch, output_dim)`` tensor through it is the identity, which
    makes the downstream embedding math exactly predictable.

    :param output_dim: Dimensionality reported to the projection head. Inputs
        must flatten to exactly this many features per sample.
    """

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten every sample to a ``(batch, output_dim)`` tensor.

        :param x: Input tensor of any shape with a leading batch axis.
        :return: The flattened input.
        """
        return x.reshape(x.shape[0], -1)


class MeanBackbone(torch.nn.Module):
    """Backbone stub reducing an image batch to per-channel spatial means.

    Accepts standard ``(batch, 3, H, W)`` image tensors of any spatial size,
    so it can stand in for ResNet-18 in image-level tests without weights.
    """

    output_dim = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average over the spatial axes.

        :param x: Image tensor of shape ``(batch, channels, H, W)``.
        :return: Tensor of shape ``(batch, channels)``.
        """
        return x.mean(dim=(2, 3))


class FakeFlowerDataset:
    """In-memory stand-in for :class:`OxfordFlowerDataset`.

    Mirrors the ``(image, label, path)`` triple interface without touching the
    filesystem or network.

    :param images: One ``uint8`` HWC array per item.
    :param labels: One integer class label per item.
    """

    def __init__(self, images: list[UInt8NumpyArray], labels: list[int]) -> None:
        assert len(images) == len(labels)
        self._images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[UInt8NumpyArray, int, str]:
        return self._images[idx], self.labels[idx], f"/fake/img_{idx}.jpg"


def build_stub_model(
    backbone_module: torch.nn.Module, **kwargs: object
) -> SiameseNeuralNetwork:
    """Build a :class:`SiameseNeuralNetwork` running on a stub backbone.

    ``SiameseNeuralNetwork`` only accepts backbone *names* and resolves them
    to pretrained networks, so the resolver is patched to return the given
    stub during construction. All other constructor keywords pass through.

    :param backbone_module: Module to install as the model's backbone.
    :param kwargs: Extra keyword arguments for ``SiameseNeuralNetwork``.
    :return: A model whose backbone is ``backbone_module``.
    """
    with mock.patch.object(
        SiameseNeuralNetwork,
        "_get_backbone",
        staticmethod(lambda name, *args, **kwargs: backbone_module),
    ):
        return SiameseNeuralNetwork(**kwargs)  # type: ignore[arg-type]


def install_head(model: SiameseNeuralNetwork, head: torch.nn.Module) -> None:
    """Install a projection head on the model's private ``_head`` attribute.

    The public ``head`` property is read-only by design (see the dedicated
    read-only property tests), so tests that need a controlled head install
    it directly on the private attribute.

    :param model: The model to modify.
    :param head: The replacement projection head.
    """
    model._head = head.to(model.device)


def make_identity_head(dim: int) -> torch.nn.Linear:
    """Build a ``Linear(dim, dim)`` head that is exactly the identity map.

    :param dim: Input and output dimensionality of the head.
    :return: A linear layer with identity weights and zero bias.
    """
    head = torch.nn.Linear(dim, dim)
    with torch.no_grad():
        head.weight.copy_(torch.eye(dim))
        head.bias.zero_()
    return head


def make_solid_rgb_image(value: int, size: int = 32) -> UInt8NumpyArray:
    """Create a uniform RGB test image.

    :param value: Pixel intensity for all channels (0-255).
    :param size: Height and width of the square image.
    :return: A ``(size, size, 3)`` ``uint8`` array.
    """
    return np.full((size, size, 3), value, dtype=np.uint8)


def make_random_rgb_image(seed: int, size: int = 32) -> UInt8NumpyArray:
    """Create a reproducible random RGB test image.

    :param seed: Seed controlling the pixel pattern.
    :param size: Height and width of the square image.
    :return: A ``(size, size, 3)`` ``uint8`` array.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def oxford_flowers_cached() -> bool:
    """Check whether the Oxford Flowers data is already on disk.

    Used to guard integration tests so the fast suite never triggers the
    ~330 MB dataset download in environments without the cache.

    :return: ``True`` if the dataset is downloaded and passes integrity checks.
    """
    from pyvisim.datasets.datasets import _check_data_integrity, _data_downloaded

    return _data_downloaded() and _check_data_integrity()


def resnet18_weights_cached() -> bool:
    """Check whether the torchvision ResNet-18 weights are already on disk.

    :return: ``True`` if a cached ``resnet18-*.pth`` checkpoint exists.
    """
    checkpoints = os.path.join(torch.hub.get_dir(), "checkpoints")
    if not os.path.isdir(checkpoints):
        return False
    return any(
        name.startswith("resnet18-") and name.endswith(".pth")
        for name in os.listdir(checkpoints)
    )
