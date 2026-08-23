"""Tests for the backbones and their preprocessing registry."""

from __future__ import annotations

import pytest
import torch
from torchvision import transforms

from pyvisim.neural_networks.siamese import ResNetBackbone, get_transform


@pytest.fixture(scope="module")
def backbone() -> ResNetBackbone:
    """A randomly initialised ResNet-18 backbone (no weight download).

    :return: A ``ResNetBackbone`` with ``pretrained=False``.
    """
    torch.manual_seed(0)
    return ResNetBackbone(pretrained=False)


def test_output_dim_is_512(backbone: ResNetBackbone) -> None:
    """ResNet-18's feature dimensionality is reported as 512."""
    assert backbone.output_dim == 512


def test_forward_shape_matches_output_dim(backbone: ResNetBackbone) -> None:
    """A ``(B, 3, 224, 224)`` batch maps to a flat ``(B, output_dim)`` tensor."""
    features = backbone(torch.randn(2, 3, 224, 224))
    assert features.shape == (2, backbone.output_dim)


def test_forward_output_is_flattened(backbone: ResNetBackbone) -> None:
    """The classifier head is stripped and the output carries no spatial axes."""
    features = backbone(torch.randn(1, 3, 224, 224))
    assert features.ndim == 2


def test_forward_accepts_other_spatial_sizes(backbone: ResNetBackbone) -> None:
    """Adaptive pooling makes the backbone size-agnostic (e.g. 64x64 inputs)."""
    features = backbone(torch.randn(1, 3, 64, 64))
    assert features.shape == (1, backbone.output_dim)


def test_no_classification_layer_remains(backbone: ResNetBackbone) -> None:
    """The final fully-connected layer of ResNet-18 is removed."""
    assert not any(
        isinstance(module, torch.nn.Linear) for module in backbone.features.modules()
    )


def test_eval_forward_is_deterministic(backbone: ResNetBackbone) -> None:
    """In eval mode, identical inputs produce identical features."""
    backbone.eval()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        first = backbone(x)
        second = backbone(x)
    assert torch.equal(first, second)


def test_pretrained_backbone_loads_imagenet_weights() -> None:
    """``pretrained=True`` loads the ImageNet weights and keeps the interface.

    The weights are downloaded on first use and cached by ``torch.hub``.
    """
    backbone = ResNetBackbone(pretrained=True)
    features = backbone(torch.randn(1, 3, 224, 224))
    assert features.shape == (1, 512)


# The preprocessing registry


def test_get_transform_of_resnet18_is_the_imagenet_pipeline() -> None:
    """The resnet18 transform is the documented ImageNet pipeline.

    Resize to 256, center-crop to 224, convert to tensor and normalize with
    the ImageNet channel statistics.
    """
    steps = get_transform("resnet18").transforms
    assert [type(step) for step in steps] == [
        transforms.Resize,
        transforms.CenterCrop,
        transforms.ToTensor,
        transforms.Normalize,
    ]
    assert steps[0].size == 256
    assert steps[1].size == (224, 224)
    assert steps[3].mean == [0.485, 0.456, 0.406]
    assert steps[3].std == [0.229, 0.224, 0.225]


def test_get_transform_returns_a_fresh_transform() -> None:
    """Each call builds its own transform, so no model shares one."""
    assert get_transform("resnet18") is not get_transform("resnet18")


def test_get_transform_of_unknown_backbone_raises() -> None:
    """Requesting the transform of an unregistered backbone raises."""
    with pytest.raises(ValueError, match="Unsupported backbone"):
        get_transform("alexnet")
