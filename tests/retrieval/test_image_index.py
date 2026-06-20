"""Tests for the FAISS-backed image indexes in :mod:`pyvisim.retrieval.index`."""

from __future__ import annotations

import numpy as np
import pytest

from pyvisim.retrieval.index import (
    ImageIndex,
    ImageIndexHNSW,
    ImageIndexIVFFlat,
    ImageIndexIVFPQ,
    ImageIndexScalarQuantizer,
)
from pyvisim.typing import Float32NumpyArray


def _random_gallery(
    num: int, dim: int, seed: int = 0
) -> tuple[list[str], Float32NumpyArray]:
    """Build a random in-memory gallery of paths and embedding vectors.

    :param num: Number of gallery vectors.
    :param dim: Dimensionality of each vector.
    :param seed: Seed for reproducible vectors.
    :returns: A ``(paths, vectors)`` pair not backed by any image files.
    """
    rng = np.random.default_rng(seed)
    paths = [f"img_{i}.png" for i in range(num)]
    vectors = rng.random((num, dim)).astype(np.float32)
    return paths, vectors


# Construction and exposed state


def test_is_subclass_of_image_index() -> None:
    """Both concrete indexes derive from the :class:`ImageIndex` interface."""
    assert issubclass(ImageIndexIVFFlat, ImageIndex)
    assert issubclass(ImageIndexIVFPQ, ImageIndex)


def test_len_dim_and_paths_match_insertion_order() -> None:
    """The index reports the gallery size, dimensionality and ordered paths."""
    paths, vectors = _random_gallery(num=40, dim=8)
    index = ImageIndexIVFFlat(paths, vectors, nlist=4, nprobe=2)
    assert len(index) == 40
    assert index.dim == 8
    assert index.paths == paths


# Search


def test_search_returns_expected_shapes() -> None:
    """``search`` returns ``(M, k)`` score and id arrays."""
    paths, vectors = _random_gallery(num=40, dim=8)
    index = ImageIndexIVFFlat(paths, vectors, nlist=4, nprobe=4)
    scores, ids = index.search(vectors[:3], k=5)
    assert scores.shape == (3, 5)
    assert ids.shape == (3, 5)


def test_search_accepts_single_vector() -> None:
    """A 1-D query vector is treated as a single-row batch."""
    paths, vectors = _random_gallery(num=40, dim=8)
    index = ImageIndexIVFFlat(paths, vectors, nlist=4, nprobe=4)
    scores, ids = index.search(vectors[0], k=3)
    assert scores.shape == (1, 3)
    assert ids.shape == (1, 3)


@pytest.mark.parametrize("quantizer", ["l2", "inner_product"])
def test_exhaustive_search_recovers_self(quantizer: str) -> None:
    """With ``nprobe == nlist`` each gallery vector is its own top match."""
    paths, vectors = _random_gallery(num=40, dim=8)
    index = ImageIndexIVFFlat(paths, vectors, quantizer=quantizer, nlist=4, nprobe=4)
    _, ids = index.search(vectors, k=1)
    assert ids[:, 0].tolist() == list(range(40))


def test_search_ids_index_into_paths() -> None:
    """Every returned id is a valid position into :attr:`paths`."""
    paths, vectors = _random_gallery(num=40, dim=8)
    index = ImageIndexIVFFlat(paths, vectors, nlist=4, nprobe=2)
    _, ids = index.search(vectors[:5], k=3)
    valid = ids[ids >= 0]
    assert valid.min() >= 0
    assert valid.max() < len(index)


def test_search_rejects_non_positive_k() -> None:
    """A non-positive ``k`` is rejected."""
    paths, vectors = _random_gallery(num=20, dim=8)
    index = ImageIndexIVFFlat(paths, vectors, nlist=2)
    with pytest.raises(ValueError, match="'k' must be a positive integer"):
        index.search(np.zeros((1, 8), dtype=np.float32), k=0)


# Metric / normalization behaviour


def test_inner_product_normalizes_gallery_vectors() -> None:
    """Inner-product indexes L2-normalise the gallery before indexing."""
    paths, vectors = _random_gallery(num=40, dim=8)
    index = ImageIndexIVFFlat(paths, vectors, quantizer="inner_product", nlist=4)
    norms = np.linalg.norm(index.reconstruct(), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_inner_product_does_not_mutate_caller_vectors() -> None:
    """Building an inner-product index leaves the caller's matrix untouched."""
    paths, vectors = _random_gallery(num=40, dim=8)
    original = vectors.copy()
    ImageIndexIVFFlat(paths, vectors, quantizer="inner_product", nlist=4)
    assert np.array_equal(vectors, original)


def test_l2_leaves_gallery_vectors_unnormalized() -> None:
    """L2 indexes keep the raw gallery vectors."""
    paths, vectors = _random_gallery(num=40, dim=8)
    index = ImageIndexIVFFlat(paths, vectors, quantizer="l2", nlist=4)
    norms = np.linalg.norm(index.reconstruct(), axis=1)
    assert not np.allclose(norms, 1.0)


def test_reconstruct_recovers_l2_gallery_exactly() -> None:
    """An IVF-Flat L2 index reconstructs the original gallery vectors exactly."""
    paths, vectors = _random_gallery(num=40, dim=8)
    index = ImageIndexIVFFlat(paths, vectors, quantizer="l2", nlist=4)
    assert np.allclose(index.reconstruct(), vectors, atol=1e-6)


# Cluster centers


def test_cluster_centers_shape() -> None:
    """``cluster_centers`` returns one centroid per cell."""
    paths, vectors = _random_gallery(num=40, dim=8)
    index = ImageIndexIVFFlat(paths, vectors, nlist=4)
    assert index.cluster_centers.shape == (4, 8)


# Validation


def test_unknown_quantizer_raises() -> None:
    """An unsupported quantizer is rejected."""
    paths, vectors = _random_gallery(num=20, dim=8)
    with pytest.raises(ValueError, match="Unsupported quantizer"):
        ImageIndexIVFFlat(paths, vectors, quantizer="bogus")  # type: ignore[arg-type]


def test_empty_gallery_raises() -> None:
    """An empty gallery cannot be indexed."""
    with pytest.raises(ValueError, match="empty gallery"):
        ImageIndexIVFFlat([], np.empty((0, 8), dtype=np.float32), nlist=1)


def test_paths_vectors_length_mismatch_raises() -> None:
    """The number of paths must match the number of gallery vectors."""
    _, vectors = _random_gallery(num=20, dim=8)
    with pytest.raises(ValueError, match="must match the number of"):
        ImageIndexIVFFlat(["only_one.png"], vectors, nlist=2)


def test_nlist_larger_than_gallery_raises() -> None:
    """``nlist`` may not exceed the number of indexed vectors."""
    paths, vectors = _random_gallery(num=20, dim=8)
    with pytest.raises(ValueError, match="'nlist' must be between 1"):
        ImageIndexIVFFlat(paths, vectors, nlist=999)


def test_nprobe_out_of_range_raises() -> None:
    """``nprobe`` must lie within ``[1, nlist]``."""
    paths, vectors = _random_gallery(num=20, dim=8)
    with pytest.raises(ValueError, match="'nprobe' must be between 1"):
        ImageIndexIVFFlat(paths, vectors, nlist=4, nprobe=99)


# IVF-PQ specifics


def test_ivfpq_search_and_cluster_centers() -> None:
    """IVF-PQ builds, searches and exposes its coarse centroids."""
    paths, vectors = _random_gallery(num=256, dim=16)
    index = ImageIndexIVFPQ(paths, vectors, nlist=4, nprobe=4, m=4, nbits=2)
    scores, ids = index.search(vectors[:3], k=5)
    assert scores.shape == (3, 5)
    assert ids.shape == (3, 5)
    assert index.cluster_centers.shape == (4, 16)
    assert (index.m, index.nbits) == (4, 2)


def test_ivfpq_m_must_divide_dim_raises() -> None:
    """``m`` must divide the feature dimensionality."""
    paths, vectors = _random_gallery(num=64, dim=16)
    with pytest.raises(ValueError, match="must divide the vector dimensionality"):
        ImageIndexIVFPQ(paths, vectors, nlist=4, m=5, nbits=4)


def test_ivfpq_too_few_vectors_raises() -> None:
    """The gallery must hold at least ``2 ** nbits`` vectors to train PQ."""
    paths, vectors = _random_gallery(num=20, dim=16)
    with pytest.raises(ValueError, match="needs at least"):
        ImageIndexIVFPQ(paths, vectors, nlist=4, m=4, nbits=8)


# Sketched (not-yet-implemented) index structures


@pytest.mark.parametrize("index_cls", [ImageIndexHNSW, ImageIndexScalarQuantizer])
def test_sketched_indexes_raise_not_implemented(
    index_cls: type[ImageIndex],
) -> None:
    """The sketched index structures raise ``NotImplementedError`` on build."""
    paths, vectors = _random_gallery(num=20, dim=8)
    with pytest.raises(NotImplementedError):
        index_cls(paths, vectors)
