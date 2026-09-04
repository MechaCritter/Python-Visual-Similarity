"""Tests for :func:`pyvisim.utils.image_utils.iter_image_batches`."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from pyvisim.utils.image_utils import iter_image_batches


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
