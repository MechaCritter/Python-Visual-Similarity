"""Tests for the ``batch_size`` contract every similarity metric inherits."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyvisim._base_classes import SimilarityMetric
from pyvisim.typing import FloatNumpyArray, ImageInput

#: Every value the shared validator has to reject.
INVALID_BATCH_SIZES = [0, -2, -100, 1.0, "4", None]


class _ConstantMetric(SimilarityMetric):
    """A metric that scores every pair with a constant, to test the base only."""

    def similarity_score(
        self,
        images1: ImageInput,
        images2: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        return np.ones((1, 1), dtype=np.float32)


# Reading the batch size


def test_batch_size_is_taken_from_the_constructor() -> None:
    """The constructor argument reaches the exposed attribute."""
    assert _ConstantMetric(batch_size=8).batch_size == 8


def test_the_whole_input_as_one_batch_is_a_valid_batch_size() -> None:
    """``-1`` is the documented way to ask for a single batch."""
    assert _ConstantMetric(batch_size=-1).batch_size == -1


# Setting the batch size


def test_set_batch_size_replaces_the_configured_value() -> None:
    """``set_batch_size`` is the documented way to change the batch size."""
    metric = _ConstantMetric(batch_size=8)
    metric.set_batch_size(2)
    assert metric.batch_size == 2


def test_batch_size_property_is_writable() -> None:
    """Assigning the property is equivalent to calling the setter method."""
    metric = _ConstantMetric()
    metric.batch_size = 3
    assert metric.batch_size == 3


# Validation


@pytest.mark.parametrize("batch_size", INVALID_BATCH_SIZES)
def test_constructor_rejects_invalid_batch_size(batch_size: Any) -> None:
    """Only ``-1`` and positive integers describe a batch."""
    with pytest.raises(ValueError, match="batch_size"):
        _ConstantMetric(batch_size=batch_size)


@pytest.mark.parametrize("batch_size", INVALID_BATCH_SIZES)
def test_set_batch_size_rejects_invalid_batch_size(batch_size: Any) -> None:
    """The setter validates like the constructor and keeps the old value."""
    metric = _ConstantMetric(batch_size=4)
    with pytest.raises(ValueError, match="batch_size"):
        metric.set_batch_size(batch_size)
    assert metric.batch_size == 4


@pytest.mark.parametrize("batch_size", INVALID_BATCH_SIZES)
def test_batch_size_property_rejects_invalid_batch_size(batch_size: Any) -> None:
    """Assigning the property goes through the same validation."""
    metric = _ConstantMetric(batch_size=4)
    with pytest.raises(ValueError, match="batch_size"):
        metric.batch_size = batch_size
    assert metric.batch_size == 4
