"""Tests for :meth:`pyvisim._base_classes.FeatureExtractorBase.extract_batch`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyvisim.features import RootSIFT

if TYPE_CHECKING:
    from tests.conftest import ImageObj


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
