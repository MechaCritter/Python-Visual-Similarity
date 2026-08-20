"""Tests for :class:`pyvisim.classic.Pipeline`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from pyvisim.classic import FisherVectorEmbedder, Pipeline, VLADEmbedder

if TYPE_CHECKING:
    from tests.conftest import ImageObj

#: Combined pipeline width: VLAD (``8 * 128``) concatenated with Fisher
#: (``2 * 8 * 128 + 8``).
PIPELINE_DIM = 8 * 128 + (2 * 8 * 128 + 8)


@pytest.fixture(scope="module")
def pipeline_embedders(
    category_train_images_flat: list[np.ndarray],
) -> tuple[VLADEmbedder, FisherVectorEmbedder]:
    """A learned VLAD and Fisher embedder, both sharing the default RootSIFT.

    :param category_train_images_flat: flattened training images to learn from.
    :returns: a ``(vlad, fisher)`` tuple of fitted embedders.
    """
    vlad = VLADEmbedder(n_clusters=8, kmeans_params={"rng": 0})
    fisher = FisherVectorEmbedder(n_components=8, gmm_params={"rng": 0})
    vlad.learn(category_train_images_flat)
    fisher.learn(category_train_images_flat)
    return vlad, fisher


@pytest.fixture(scope="module")
def pipeline(
    pipeline_embedders: tuple[VLADEmbedder, FisherVectorEmbedder],
) -> Pipeline:
    """A pipeline combining the learned VLAD and Fisher embedders.

    :param pipeline_embedders: the ``(vlad, fisher)`` fixture.
    :returns: a ``Pipeline`` over both embedders.
    """
    vlad, fisher = pipeline_embedders
    return Pipeline([vlad, fisher])


def test_rejects_non_embedder() -> None:
    """A pipeline rejects members that are not embedders."""
    with pytest.raises(
        ValueError, match="only accepts instances of SerializableImageEmbedder"
    ):
        Pipeline(["not an embedder"])  # type: ignore[list-item]


def test_embed_concatenates(pipeline: Pipeline, checkerboard_image: ImageObj) -> None:
    """The pipeline concatenates each embedder's output along the feature axis."""
    out = pipeline.embed([checkerboard_image.array])
    assert out.shape == (1, PIPELINE_DIM)


def test_restores_flatten_flag(
    pipeline: Pipeline,
    pipeline_embedders: tuple[VLADEmbedder, FisherVectorEmbedder],
    checkerboard_image: ImageObj,
) -> None:
    """The pipeline restores each embedder's original ``flatten`` flag."""
    vlad, _ = pipeline_embedders
    original = vlad.flatten
    vlad.flatten = False
    try:
        pipeline.embed([checkerboard_image.array])
        assert vlad.flatten is False
    finally:
        vlad.flatten = original


def test_embed_batch(pipeline: Pipeline, checkerboard_image: ImageObj) -> None:
    """A batch of two images embeds to ``(2, PIPELINE_DIM)``."""
    base = checkerboard_image.array
    batch = [base, np.roll(base, 8, axis=0)]
    assert pipeline.embed(batch).shape == (2, PIPELINE_DIM)


def test_embed_accepts_tensor(pipeline: Pipeline, checkerboard_image: ImageObj) -> None:
    """A grayscale torch tensor image is accepted and embeds like its array."""
    tensor = torch.from_numpy(checkerboard_image.array)
    assert pipeline.embed([tensor]).shape == (1, PIPELINE_DIM)


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
    """``repr`` names the pipeline and each member embedder."""
    text = repr(pipeline)
    assert "Pipeline(" in text
    assert "VLADEmbedder(" in text
    assert "FisherVectorEmbedder(" in text


def test_to_dict_from_dict_round_trip(
    pipeline: Pipeline, checkerboard_image: ImageObj
) -> None:
    """``Pipeline.from_dict(pipeline.to_dict())`` rebuilds an equivalent pipeline."""
    restored = Pipeline.from_dict(pipeline.to_dict())
    assert isinstance(restored, Pipeline)
    assert len(restored.embedders) == len(pipeline.embedders)
    assert restored.similarity_func_name == pipeline.similarity_func_name
    probe = [checkerboard_image.array]
    assert np.allclose(restored.embed(probe), pipeline.embed(probe), atol=1e-5)
