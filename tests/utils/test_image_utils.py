"""Tests for :mod:`pyvisim.utils.image_utils`."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from pyvisim._errors import InvalidImageError
from pyvisim.utils.image_utils import iter_image_batches, to_single_image


def _images(count: int) -> list[np.ndarray]:
    """Build ``count`` distinguishable 4x4 grayscale images.

    :param count: how many images to build.
    :returns: a list of ``uint8`` images.
    """
    return [np.full((4, 4), fill, dtype=np.uint8) for fill in range(count)]


def test_batches_have_the_requested_size() -> None:
    """The input is split into full batches, the last one possibly shorter."""
    batches = list(iter_image_batches(_images(5), 2))
    assert [len(batch) for batch in batches] == [2, 2, 1]


def test_batch_size_minus_one_yields_one_batch() -> None:
    """``-1`` hands the whole input over in a single batch."""
    batches = list(iter_image_batches(_images(5), -1))
    assert [len(batch) for batch in batches] == [5]


def test_batches_preserve_the_input_order() -> None:
    """Regrouping images must not reorder them."""
    images = _images(5)
    flattened = [image for batch in iter_image_batches(images, 2) for image in batch]
    for original, yielded in zip(images, flattened, strict=True):
        np.testing.assert_array_equal(original, yielded)


@pytest.mark.parametrize("batch_size", [1, 3, -1])
def test_an_empty_input_yields_no_batch(batch_size: int) -> None:
    """An input holding no image yields no batch at all, not an empty one."""
    assert list(iter_image_batches([], batch_size)) == []


def test_a_generator_is_consumed_one_batch_at_a_time() -> None:
    """Batching stays lazy, so a large stream is never materialized at once."""
    consumed = 0

    def source() -> Iterator[np.ndarray]:
        nonlocal consumed
        for image in _images(6):
            consumed += 1
            yield image

    batches = iter_image_batches(source(), 2)
    next(batches)
    assert consumed == 2


def test_a_batched_array_is_split_by_its_dims() -> None:
    """A stacked array is expanded into its images before being batched."""
    stacked = np.stack(_images(4))
    assert [len(batch) for batch in iter_image_batches(stacked, 3, dims="BHW")] == [
        3,
        1,
    ]


@pytest.mark.parametrize("batch_size", [0, -2])
def test_invalid_batch_size_raises(batch_size: int) -> None:
    """Only ``-1`` and positive integers describe a batch."""
    with pytest.raises(ValueError, match="batch_size"):
        list(iter_image_batches(_images(2), batch_size))


def test_to_single_image_reorders_chw_into_hwc() -> None:
    """A channels-first image comes back in the canonical ``(H, W, C)`` layout."""
    image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    np.testing.assert_array_equal(
        to_single_image(image.transpose(2, 0, 1), dims="CHW"), image
    )


def test_to_single_image_keeps_a_grayscale_image_two_dimensional() -> None:
    """A 2-D image with the channel-bearing ``"HWC"`` is read as grayscale."""
    image = _images(1)[0]
    np.testing.assert_array_equal(to_single_image(image), image)


def test_to_single_image_rescales_the_value_range() -> None:
    """Values in ``value_range`` are mapped onto ``[0, 255]`` as ``uint8``."""
    image = np.array([[0.0, 0.5, 1.0]])
    result = to_single_image(image, dims="HW", value_range=(0.0, 1.0))
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, [[0, 127, 255]])


def test_to_single_image_accepts_a_batch_of_one() -> None:
    """A batch axis holding exactly one image is unwrapped."""
    image = _images(1)[0]
    np.testing.assert_array_equal(to_single_image(image[np.newaxis], dims="BHW"), image)


def test_to_single_image_rejects_several_images() -> None:
    """A batch holding more than one image is not a single image."""
    with pytest.raises(ValueError, match="single image"):
        to_single_image(np.stack(_images(2)), dims="BHW")


def test_to_single_image_rejects_non_numeric_input() -> None:
    """A string is not an image and raises ``InvalidImageError``."""
    with pytest.raises(InvalidImageError):
        to_single_image("not an image", dims="HW")
