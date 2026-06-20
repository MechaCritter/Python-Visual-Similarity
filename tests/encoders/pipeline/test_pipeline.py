"""Tests for :class:`pyvisim.encoders.Pipeline`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from pyvisim.encoders import FisherVectorEncoder, Pipeline, VLADEncoder

if TYPE_CHECKING:
    from tests.conftest import ImageObj

#: Combined pipeline width: VLAD (``8 * 128``) concatenated with Fisher
#: (``2 * 8 * 128 + 8``).
PIPELINE_DIM = 8 * 128 + (2 * 8 * 128 + 8)


@pytest.fixture(scope="module")
def pipeline_encoders(
    category_train_images_flat: list[np.ndarray],
) -> tuple[VLADEncoder, FisherVectorEncoder]:
    """A learned VLAD and Fisher encoder, both sharing the default RootSIFT.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a ``(vlad, fisher)`` tuple of fitted encoders.
    """
    vlad = VLADEncoder(n_clusters=8, kmeans_params={"random_state": 0, "n_init": 3})
    fisher = FisherVectorEncoder(n_components=8, gmm_params={"random_state": 0})
    vlad.learn(category_train_images_flat)
    fisher.learn(category_train_images_flat)
    return vlad, fisher


@pytest.fixture(scope="module")
def pipeline(
    pipeline_encoders: tuple[VLADEncoder, FisherVectorEncoder],
) -> Pipeline:
    """A pipeline combining the learned VLAD and Fisher encoders.

    :param pipeline_encoders: the ``(vlad, fisher)`` fixture.
    :returns: a ``Pipeline`` over both encoders.
    """
    vlad, fisher = pipeline_encoders
    return Pipeline([vlad, fisher])


def test_rejects_non_encoder() -> None:
    """A pipeline rejects members that are not encoders."""
    with pytest.raises(ValueError, match="only accepts instances of ImageEncoderBase"):
        Pipeline(["not an encoder"])  # type: ignore[list-item]


def test_encode_concatenates(pipeline: Pipeline, checkerboard_image: ImageObj) -> None:
    """The pipeline concatenates each encoder's output along the feature axis."""
    out = pipeline.encode([checkerboard_image.array])
    assert out.shape == (1, PIPELINE_DIM)


def test_restores_flatten_flag(
    pipeline: Pipeline,
    pipeline_encoders: tuple[VLADEncoder, FisherVectorEncoder],
    checkerboard_image: ImageObj,
) -> None:
    """The pipeline restores each encoder's original ``flatten`` flag."""
    vlad, _ = pipeline_encoders
    original = vlad.flatten
    vlad.flatten = False
    try:
        pipeline.encode([checkerboard_image.array])
        assert vlad.flatten is False
    finally:
        vlad.flatten = original


def test_encode_batch(pipeline: Pipeline, checkerboard_image: ImageObj) -> None:
    """A batch of two images encodes to ``(2, PIPELINE_DIM)``."""
    base = checkerboard_image.array
    batch = [base, np.roll(base, 8, axis=0)]
    assert pipeline.encode(batch).shape == (2, PIPELINE_DIM)


def test_encode_accepts_tensor(
    pipeline: Pipeline, checkerboard_image: ImageObj
) -> None:
    """A grayscale torch tensor image is accepted and encodes like its array."""
    tensor = torch.from_numpy(checkerboard_image.array)
    assert pipeline.encode([tensor]).shape == (1, PIPELINE_DIM)


def test_similarity_score_shape(
    pipeline: Pipeline, checkerboard_image: ImageObj
) -> None:
    """``similarity_score`` returns an ``(n1, n2)`` float32 matrix."""
    base = checkerboard_image.array
    images1 = [base, np.roll(base, 8, axis=0)]
    images2 = [np.roll(base, 8, axis=1)]
    scores = pipeline.similarity_score(images1, images2)
    assert scores.shape == (2, 1)
    assert scores.dtype == np.float32


def test_repr(pipeline: Pipeline) -> None:
    """``repr`` names the pipeline and each member encoder."""
    text = repr(pipeline)
    assert "Pipeline(" in text
    assert "VLADEncoder(" in text
    assert "FisherVectorEncoder(" in text


def test_to_dict_from_dict_round_trip(
    pipeline: Pipeline, checkerboard_image: ImageObj
) -> None:
    """``Pipeline.from_dict(pipeline.to_dict())`` rebuilds an equivalent pipeline."""
    restored = Pipeline.from_dict(pipeline.to_dict())
    assert isinstance(restored, Pipeline)
    assert len(restored.encoders) == len(pipeline.encoders)
    assert restored.similarity_func_name == pipeline.similarity_func_name
    probe = [checkerboard_image.array]
    assert np.allclose(restored.encode(probe), pipeline.encode(probe), atol=1e-5)
