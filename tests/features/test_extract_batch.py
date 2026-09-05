"""Tests for :meth:`pyvisim._base_classes.FeatureExtractorBase.extract_batch`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch.nn as nn
from torchvision import transforms

from pyvisim.features import DeepConvFeature, RootSIFT

if TYPE_CHECKING:
    from tests.conftest import ImageObj


def _tiny_conv_model() -> nn.Module:
    """Build a tiny two-conv-layer network to avoid downloading VGG16.

    :returns: a ``torch.nn.Sequential`` with two ``Conv2d`` layers (6 then 10
        output channels) separated by a ReLU.
    """
    return nn.Sequential(
        nn.Conv2d(3, 6, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(6, 10, 3, padding=1),
    )


def _rgb_images(*sizes: tuple[int, int]) -> list[np.ndarray]:
    """Build one RGB noise image per requested ``(height, width)``.

    The conv backbones expect three channels, and a batch only stacks when the
    transform brings the images to a common size.

    :param sizes: the ``(height, width)`` of each image to build.
    :returns: a list of ``(H, W, 3)`` ``uint8`` images.
    """
    rng = np.random.default_rng(0)
    return [
        rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        for height, width in sizes
    ]


def test_extract_batch_matches_one_call_per_image(
    checkerboard_image: ImageObj, stripes_image: ImageObj
) -> None:
    """The default implementation extracts each image exactly as ``__call__`` does."""
    extractor = RootSIFT()
    images = [checkerboard_image.array, stripes_image.array]
    batched = extractor.extract_batch(images)
    assert len(batched) == len(images)
    for features, image in zip(batched, images, strict=True):
        np.testing.assert_array_equal(features, extractor(image))


def test_extract_batch_keeps_one_array_per_image(
    checkerboard_image: ImageObj, stripes_image: ImageObj, solid_image: ImageObj
) -> None:
    """Images yield different descriptor counts, so the arrays stay separate."""
    extractor = RootSIFT()
    batched = extractor.extract_batch(
        [checkerboard_image.array, stripes_image.array, solid_image.array]
    )
    assert [features.shape[1] for features in batched] == [extractor.output_dim] * 3
    # A featureless image contributes an empty (0, D) array rather than nothing.
    assert len(batched[2]) == 0


def test_extract_batch_forwards_dims_and_value_range(
    checkerboard_image: ImageObj,
) -> None:
    """The layout and range arguments reach every image of the batch."""
    extractor = RootSIFT()
    scaled = checkerboard_image.array.astype(np.float32) / 255.0
    batched = extractor.extract_batch([scaled], value_range=(0.0, 1.0))
    np.testing.assert_array_equal(batched[0], extractor(checkerboard_image.array))


def test_extract_batch_of_nothing_is_empty() -> None:
    """An empty batch produces an empty list, not an error."""
    assert RootSIFT().extract_batch([]) == []


# DeepConvFeature, which pushes the whole batch through the model at once


def test_deepconv_batch_matches_one_call_per_image() -> None:
    """One forward pass over the batch yields exactly the per-image descriptors."""
    extractor = DeepConvFeature(
        _tiny_conv_model(),
        layer_index=-1,
        device="cpu",
    )
    images = _rgb_images((64, 64), (96, 80), (64, 64))
    batched = extractor.extract_batch(images)
    assert len(batched) == len(images)
    for features, image in zip(batched, images, strict=True):
        np.testing.assert_array_equal(features, extractor(image))


def test_deepconv_batch_keeps_the_output_contract() -> None:
    """Batched descriptors keep the ``(Hf * Wf, output_dim)`` float32 layout."""
    extractor = DeepConvFeature(_tiny_conv_model(), layer_index=-1, device="cpu")
    features = extractor.extract_batch(_rgb_images((64, 64), (64, 64)))[0]
    assert features.shape == (224 * 224, extractor.output_dim)
    assert features.dtype == np.float32


def test_deepconv_falls_back_when_the_batch_cannot_be_stacked() -> None:
    """A transform that keeps the input size leaves the images different shapes."""
    extractor = DeepConvFeature(
        _tiny_conv_model(),
        layer_index=-1,
        device="cpu",
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    images = _rgb_images((32, 32), (48, 40))
    batched = extractor.extract_batch(images)
    assert len({features.shape for features in batched}) == 2
    for features, image in zip(batched, images, strict=True):
        np.testing.assert_array_equal(features, extractor(image))
