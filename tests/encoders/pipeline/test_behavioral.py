"""Behavioural / requirement-driven tests for :class:`pyvisim.encoders.Pipeline`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyvisim.encoders import FisherVectorEncoder, Pipeline, VLADEncoder

if TYPE_CHECKING:
    from pyvisim._base_classes import SimilarityMetric
    from tests.conftest import ImageObj

#: A small margin so ordering assertions are not decided by floating-point noise.
MARGIN = 1e-6

#: PCA variants for the pipeline behavioural test (mirrors the encoder fixtures).
PCA_VARIANTS = [None, {"n_components": 32, "random_state": 0}]


def _assert_same_category_higher(
    encoder: SimilarityMetric, query_images: dict[str, list[np.ndarray]]
) -> None:
    """Assert same-category queries score higher than different-category ones."""
    categories = list(query_images)
    same_scores: list[float] = []
    diff_scores: list[float] = []
    for index, category in enumerate(categories):
        other = categories[(index + 1) % len(categories)]
        query_a, query_b = query_images[category][0], query_images[category][1]
        query_other = query_images[other][0]
        score_same = float(encoder.similarity_score([query_a], [query_b])[0, 0])
        score_diff = float(encoder.similarity_score([query_a], [query_other])[0, 0])
        assert score_same > score_diff + MARGIN
        same_scores.append(score_same)
        diff_scores.append(score_diff)
    assert float(np.mean(same_scores)) > float(np.mean(diff_scores)) + MARGIN


def _assert_noisy_closer_to_original_than_different(
    encoder: SimilarityMetric,
    original: np.ndarray,
    noisy: np.ndarray,
    different: np.ndarray,
) -> None:
    """Assert ``similarity(original, noisy) > similarity(original, different)``."""
    score_noisy = float(encoder.similarity_score([original], [noisy])[0, 0])
    score_diff = float(encoder.similarity_score([original], [different])[0, 0])
    assert score_noisy > score_diff + MARGIN


def test_identical_image_near_perfect_similarity(
    learned_vlad_encoder: VLADEncoder,
    learned_fisher_encoder: FisherVectorEncoder,
    checkerboard_image: ImageObj,
) -> None:
    """Pipeline reports similarity of 1.0 for an image against its copy."""
    pipeline = Pipeline([learned_vlad_encoder, learned_fisher_encoder])
    image = checkerboard_image.array
    score = float(pipeline.similarity_score([image], [image.copy()])[0, 0])
    assert score == pytest.approx(1.0, abs=1e-6)


def test_noisy_closer_to_original_than_different(
    learned_vlad_encoder: VLADEncoder,
    learned_fisher_encoder: FisherVectorEncoder,
    checkerboard_image: ImageObj,
    noisy_checkerboard_image: ImageObj,
    blobs_image: ImageObj,
) -> None:
    """Pipeline: image+noise is more similar to its original than to a structurally different image."""
    pipeline = Pipeline([learned_vlad_encoder, learned_fisher_encoder])
    _assert_noisy_closer_to_original_than_different(
        pipeline,
        checkerboard_image.array,
        noisy_checkerboard_image.array,
        blobs_image.array,
    )


# §3.6.3 Pipeline behavioural


@pytest.mark.parametrize("pca_params", PCA_VARIANTS, ids=["no_pca", "pca32"])
def test_same_category_higher(
    category_train_images_flat: list[np.ndarray],
    category_query_images: dict[str, list[np.ndarray]],
    pca_params: dict[str, int] | None,
) -> None:
    """A VLAD+Fisher pipeline scores same-category pairs higher than different ones."""
    vlad = VLADEncoder(
        n_clusters=8,
        kmeans_params={"random_state": 0, "n_init": 3},
        pca_params=pca_params,
    )
    fisher = FisherVectorEncoder(
        n_components=8, gmm_params={"random_state": 0}, pca_params=pca_params
    )
    vlad.learn(category_train_images_flat)
    fisher.learn(category_train_images_flat)
    pipeline = Pipeline([vlad, fisher])
    _assert_same_category_higher(pipeline, category_query_images)
