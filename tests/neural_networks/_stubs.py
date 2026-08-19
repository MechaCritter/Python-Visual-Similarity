"""Shared stubs and helpers for the ``pyvisim.neural_networks`` test suite."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar
from unittest import mock

import numpy as np
import torch

from pyvisim.neural_networks import BCESiameseNetwork, ContrastiveSiameseNetwork
from pyvisim.neural_networks.siamese._base_siamese import SiameseNetworkBase
from pyvisim.typing import UInt8NumpyArray

_SiameseT = TypeVar("_SiameseT", bound=SiameseNetworkBase)


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
        return x.reshape(x.shape[0], -1)


class MeanBackbone(torch.nn.Module):
    """Backbone stub reducing an image batch to per-channel spatial means.

    Accepts standard ``(batch, 3, H, W)`` image tensors of any spatial size,
    so it can stand in for ResNet-18 in image-level tests without weights.
    """

    output_dim = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=(2, 3))


def _build_stub_network(
    model_cls: type[_SiameseT], backbone_module: torch.nn.Module, **kwargs: object
) -> _SiameseT:
    """Install a projection head on the model's private ``_head`` attribute.

    The public ``head`` property is read-only by design (see the dedicated
    read-only property tests), so tests that need a controlled head install
    it directly on the private attribute.

    :param model: The model to modify.
    :param head: The replacement projection head.
    """
    with mock.patch.object(
        model_cls,
        "_get_backbone",
        staticmethod(lambda name, *args, **kwargs: backbone_module),
    ):
        return model_cls(**kwargs)  # type: ignore[arg-type]


def build_stub_model(
    backbone_module: torch.nn.Module, **kwargs: object
) -> ContrastiveSiameseNetwork:
    return _build_stub_network(ContrastiveSiameseNetwork, backbone_module, **kwargs)


def build_stub_bce_model(
    backbone_module: torch.nn.Module, **kwargs: object
) -> BCESiameseNetwork:
    return _build_stub_network(BCESiameseNetwork, backbone_module, **kwargs)


def install_head(model: SiameseNetworkBase, head: torch.nn.Module) -> None:
    """Install a projection head on the model's private ``_head`` attribute.

    The public ``head`` property is read-only by design (see the dedicated
    read-only property tests), so tests that need a controlled head install
    it directly on the private attribute.
    """
    model._head = head.to(model.device)


def install_scorer(
    model: BCESiameseNetwork, weights: Sequence[float], bias: float
) -> None:
    scorer = model.scorer
    assert isinstance(scorer, torch.nn.Linear)
    with torch.no_grad():
        scorer.weight.copy_(torch.tensor([list(weights)], dtype=torch.float32))
        scorer.bias.fill_(bias)


def make_identity_head(dim: int) -> torch.nn.Linear:
    head = torch.nn.Linear(dim, dim)
    with torch.no_grad():
        head.weight.copy_(torch.eye(dim))
        head.bias.zero_()
    return head


def make_solid_rgb_image(value: int, size: int = 32) -> UInt8NumpyArray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def make_random_rgb_image(seed: int, size: int = 32) -> UInt8NumpyArray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
