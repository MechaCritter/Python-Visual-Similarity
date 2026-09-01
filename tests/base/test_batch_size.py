"""Tests for the ``batch_size`` contract shared by every similarity metric."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyvisim._base_classes import (
    ImageEmbedderBase,
    SerializableImageEmbedder,
    SimilarityMetric,
)
from pyvisim.base import DenseMetricBase
from pyvisim.serialization import save_embedder_state
from pyvisim.typing import Float64NumpyArray, FloatNumpyArray, ImageInput

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


class _ConstantEmbedder(ImageEmbedderBase):
    """An embedder that emits a constant vector, to test the base only."""

    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        return np.ones((1, 4), dtype=np.float32)


class _StubSerializableEmbedder(SerializableImageEmbedder):
    """A serializable embedder reduced to the state the base class owns."""

    def embed(
        self,
        images: ImageInput,
        *,
        dims: str = "HWC",
        value_range: tuple[float, float] = (0.0, 255.0),
    ) -> FloatNumpyArray:
        return np.ones((1, 4), dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "embedder_class": type(self).__name__,
            "similarity_func": self.similarity_func_name,
            "batch_size": self.batch_size,
        }

    @classmethod
    def from_dict(
        cls, state: dict[str, Any], **kwargs: Any
    ) -> _StubSerializableEmbedder:
        embedder = cls(similarity_func=state["similarity_func"])
        embedder._restore_batch_size(state)
        return embedder


class _ZeroDenseMetric(DenseMetricBase):
    """A dense metric that scores every pair with zero, to test the base only."""

    def _score_pairs(
        self, images1: Float64NumpyArray, images2: Float64NumpyArray
    ) -> Float64NumpyArray:
        return np.zeros(len(images1), dtype=np.float64)


def test_batch_size_defaults_to_whole_input() -> None:
    """A metric processes the whole input as one batch unless told otherwise."""
    assert _ConstantMetric().batch_size == -1


def test_batch_size_is_taken_from_the_constructor() -> None:
    """The constructor argument reaches the exposed attribute."""
    assert _ConstantMetric(batch_size=8).batch_size == 8


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


@pytest.mark.parametrize("batch_size", INVALID_BATCH_SIZES)
def test_constructor_rejects_invalid_batch_size(batch_size: Any) -> None:
    """Only ``-1`` and positive integers describe a batch."""
    with pytest.raises(ValueError, match="batch_size"):
        _ConstantMetric(batch_size=batch_size)


@pytest.mark.parametrize("batch_size", INVALID_BATCH_SIZES)
def test_set_batch_size_rejects_invalid_batch_size(batch_size: Any) -> None:
    """The setter validates exactly like the constructor."""
    metric = _ConstantMetric()
    with pytest.raises(ValueError, match="batch_size"):
        metric.set_batch_size(batch_size)
    assert metric.batch_size == -1


def test_embedders_share_the_batch_size_contract() -> None:
    """``ImageEmbedderBase`` forwards ``batch_size`` to the metric base."""
    embedder = _ConstantEmbedder(similarity_func="euclidean", batch_size=4)
    assert embedder.batch_size == 4
    assert embedder.similarity_func_name == "euclidean"
    embedder.set_batch_size(-1)
    assert embedder.batch_size == -1


def test_batch_size_survives_a_state_round_trip() -> None:
    """A serialised embedder comes back with the batch size it was saved with."""
    embedder = _StubSerializableEmbedder(batch_size=16)
    restored = _StubSerializableEmbedder.from_dict(embedder.to_dict())
    assert restored.batch_size == 16


def test_batch_size_survives_a_file_round_trip(tmp_path: Path) -> None:
    """The batch size is part of the ``.embedder`` file, not just of the state."""
    path = _StubSerializableEmbedder(batch_size=16).save_to_disk(tmp_path / "stub")
    assert _StubSerializableEmbedder.load_from_disk(path).batch_size == 16


def test_state_without_a_batch_size_is_not_a_valid_file(tmp_path: Path) -> None:
    """The batch size is a required key, so a file predating it is rejected."""
    assert "batch_size" in _StubSerializableEmbedder._STATE_KEYS
    state = _StubSerializableEmbedder().to_dict()
    del state["batch_size"]
    save_embedder_state(state, path := tmp_path / "stub.embedder")
    with pytest.raises(ValueError, match="not a valid .embedder file"):
        _StubSerializableEmbedder.load_from_disk(path)


def test_dense_metrics_share_the_batch_size_contract() -> None:
    """``DenseMetricBase`` inherits the shared attribute and setter."""
    metric = _ZeroDenseMetric(batch_size=2)
    assert metric.batch_size == 2
    metric.set_batch_size(-1)
    assert metric.batch_size == -1
    with pytest.raises(ValueError, match="batch_size"):
        _ZeroDenseMetric(batch_size=0)


def test_dense_metric_batching_does_not_change_scores() -> None:
    """Chunked pair scoring covers the same grid as one big batch."""
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, (3, 8, 8, 3), dtype=np.uint8)
    whole = _ZeroDenseMetric().similarity_score(images, images, dims="BHWC")
    chunked = _ZeroDenseMetric(batch_size=2).similarity_score(
        images, images, dims="BHWC"
    )
    assert whole.shape == (3, 3)
    np.testing.assert_array_equal(whole, chunked)
