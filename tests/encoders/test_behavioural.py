"""Behavioural / requirement-driven encoder tests.

These integration tests learn a visual vocabulary and then assert the ordering
of similarity scores. Each behaviour is checked for both encoders (via the
``learned_vlad_encoder`` / ``learned_fisher_encoder`` fixtures, which are
themselves parametrized over a non-PCA and a PCA variant), so every assertion
runs across the full encoder x PCA matrix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyvisim.encoders import FisherVectorEncoder, Pipeline, VLADEncoder

if TYPE_CHECKING:
    from pyvisim._base_classes import SimilarityMetric
    from tests.conftest import ImageObj

#: A small margin so ordering assertions are not decided by floating-point noise.
MARGIN = 1e-3

#: PCA variants for the pipeline behavioural test (mirrors the encoder fixtures).
PCA_VARIANTS = [None, {"n_components": 32, "random_state": 0}]


# Shared assertion helpers (one per behaviour, reused across both encoders)


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


# §3.6.1 Category separation


def test_same_category_scores_higher_than_different_vlad(
    learned_vlad_encoder: VLADEncoder,
    category_query_images: dict[str, list[np.ndarray]],
) -> None:
    """VLAD scores same-category query pairs higher than different-category ones."""
    _assert_same_category_higher(learned_vlad_encoder, category_query_images)


def test_same_category_scores_higher_than_different_fisher(
    learned_fisher_encoder: FisherVectorEncoder,
    category_query_images: dict[str, list[np.ndarray]],
) -> None:
    """Fisher scores same-category query pairs higher than different-category ones."""
    _assert_same_category_higher(learned_fisher_encoder, category_query_images)


def test_identical_image_near_perfect_similarity_vlad(
    learned_vlad_encoder: VLADEncoder, checkerboard_image: ImageObj
) -> None:
    """VLAD reports similarity of 1.0 for an image against its copy."""
    image = checkerboard_image.array
    score = float(learned_vlad_encoder.similarity_score([image], [image.copy()])[0, 0])
    assert score == pytest.approx(1.0, abs=1e-6)


def test_identical_image_near_perfect_similarity_fisher(
    learned_fisher_encoder: FisherVectorEncoder, checkerboard_image: ImageObj
) -> None:
    """Fisher reports similarity of 1.0 for an image against its copy."""
    image = checkerboard_image.array
    score = float(
        learned_fisher_encoder.similarity_score([image], [image.copy()])[0, 0]
    )
    assert score == pytest.approx(1.0, abs=1e-6)


def test_identical_image_near_perfect_similarity_pipeline(
    learned_vlad_encoder: VLADEncoder,
    learned_fisher_encoder: FisherVectorEncoder,
    checkerboard_image: ImageObj,
) -> None:
    """Pipeline reports similarity of 1.0 for an image against its copy."""
    pipeline = Pipeline([learned_vlad_encoder, learned_fisher_encoder])
    image = checkerboard_image.array
    score = float(pipeline.similarity_score([image], [image.copy()])[0, 0])
    assert score == pytest.approx(1.0, abs=1e-6)


def test_encode_deterministic_vlad(
    learned_vlad_encoder: VLADEncoder, checkerboard_image: ImageObj
) -> None:
    """Encoding the same query twice with a fixed seed is reproducible."""
    image = checkerboard_image.array
    assert np.allclose(
        learned_vlad_encoder.encode([image]), learned_vlad_encoder.encode([image])
    )


def test_encode_deterministic_fisher(
    learned_fisher_encoder: FisherVectorEncoder, checkerboard_image: ImageObj
) -> None:
    """Encoding the same query twice with a fixed seed is reproducible."""
    image = checkerboard_image.array
    assert np.allclose(
        learned_fisher_encoder.encode([image]), learned_fisher_encoder.encode([image])
    )


# §3.6.2 Noise-level ordering


def _assert_same_noise_higher(
    encoder: SimilarityMetric,
    noisy_pair: tuple[ImageObj, ImageObj],
    very_noisy_pair: tuple[ImageObj, ImageObj],
) -> float:
    """Assert same-level noise scores higher than different-level; return ``s_diff``."""
    noisy_a, noisy_b = noisy_pair
    very_noisy_a, _ = very_noisy_pair
    score_same = float(encoder.similarity_score([noisy_a.array], [noisy_b.array])[0, 0])
    score_diff = float(
        encoder.similarity_score([noisy_a.array], [very_noisy_a.array])[0, 0]
    )
    assert score_same > score_diff + MARGIN
    return score_diff


def _assert_grey_is_floor(
    encoder: SimilarityMetric, query: np.ndarray, grey: np.ndarray
) -> None:
    """Assert the plain grey image is the similarity floor (it cannot be encoded).

    The plain grey image yields no descriptors, so the encoders raise rather
    than producing a score: it is strictly less comparable than any encodable
    pair, i.e. ``s_diff > s_grey``.
    """
    with pytest.raises(ValueError):
        encoder.similarity_score([query], [grey])


def test_same_noise_more_similar_than_different_noise_vlad(
    learned_vlad_encoder: VLADEncoder,
    noisy_image_pair: tuple[ImageObj, ImageObj],
    very_noisy_image_pair: tuple[ImageObj, ImageObj],
) -> None:
    """VLAD: same noise level is more similar than a different noise level."""
    _assert_same_noise_higher(
        learned_vlad_encoder, noisy_image_pair, very_noisy_image_pair
    )


def test_same_noise_more_similar_than_different_noise_fisher(
    learned_fisher_encoder: FisherVectorEncoder,
    noisy_image_pair: tuple[ImageObj, ImageObj],
    very_noisy_image_pair: tuple[ImageObj, ImageObj],
) -> None:
    """Fisher: same noise level is more similar than a different noise level."""
    _assert_same_noise_higher(
        learned_fisher_encoder, noisy_image_pair, very_noisy_image_pair
    )


def test_different_noise_more_similar_than_grey_vlad(
    learned_vlad_encoder: VLADEncoder,
    noisy_image_pair: tuple[ImageObj, ImageObj],
    solid_image: ImageObj,
) -> None:
    """VLAD: a different-noise pair still beats the unencodable plain grey floor."""
    noisy_a, _ = noisy_image_pair
    _assert_grey_is_floor(learned_vlad_encoder, noisy_a.array, solid_image.array)


def test_different_noise_more_similar_than_grey_fisher(
    learned_fisher_encoder: FisherVectorEncoder,
    noisy_image_pair: tuple[ImageObj, ImageObj],
    solid_image: ImageObj,
) -> None:
    """Fisher: a different-noise pair still beats the unencodable plain grey floor."""
    noisy_a, _ = noisy_image_pair
    _assert_grey_is_floor(learned_fisher_encoder, noisy_a.array, solid_image.array)


def test_full_noise_ordering_vlad(
    learned_vlad_encoder: VLADEncoder,
    noisy_image_pair: tuple[ImageObj, ImageObj],
    very_noisy_image_pair: tuple[ImageObj, ImageObj],
    solid_image: ImageObj,
) -> None:
    """VLAD: ``s_same > s_diff > s_grey`` as a single ordering check."""
    _assert_same_noise_higher(
        learned_vlad_encoder, noisy_image_pair, very_noisy_image_pair
    )
    _assert_grey_is_floor(
        learned_vlad_encoder, noisy_image_pair[0].array, solid_image.array
    )


def test_full_noise_ordering_fisher(
    learned_fisher_encoder: FisherVectorEncoder,
    noisy_image_pair: tuple[ImageObj, ImageObj],
    very_noisy_image_pair: tuple[ImageObj, ImageObj],
    solid_image: ImageObj,
) -> None:
    """Fisher: ``s_same > s_diff > s_grey`` as a single ordering check."""
    _assert_same_noise_higher(
        learned_fisher_encoder, noisy_image_pair, very_noisy_image_pair
    )
    _assert_grey_is_floor(
        learned_fisher_encoder, noisy_image_pair[0].array, solid_image.array
    )


# §3.6.4 Noisy image closer to original than to a different image


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


def test_noisy_closer_to_original_than_different_vlad(
    learned_vlad_encoder: VLADEncoder,
    checkerboard_image: ImageObj,
    noisy_checkerboard_image: ImageObj,
    blobs_image: ImageObj,
) -> None:
    """VLAD: image+noise is more similar to its original than to a structurally different image."""
    _assert_noisy_closer_to_original_than_different(
        learned_vlad_encoder,
        checkerboard_image.array,
        noisy_checkerboard_image.array,
        blobs_image.array,
    )


def test_noisy_closer_to_original_than_different_fisher(
    learned_fisher_encoder: FisherVectorEncoder,
    checkerboard_image: ImageObj,
    noisy_checkerboard_image: ImageObj,
    blobs_image: ImageObj,
) -> None:
    """Fisher: image+noise is more similar to its original than to a structurally different image."""
    _assert_noisy_closer_to_original_than_different(
        learned_fisher_encoder,
        checkerboard_image.array,
        noisy_checkerboard_image.array,
        blobs_image.array,
    )


def test_noisy_closer_to_original_than_different_pipeline(
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
def test_pipeline_same_category_higher(
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
