"""Tests for the FAISS-backed image indexes in :mod:`pyvisim.retrieval.index`."""

from __future__ import annotations

import numpy as np
import pytest

from pyvisim.image_store import ImageEncodingMap
from pyvisim.retrieval.index import (
    ImageIndex,
    ImageIndexIVFFlat,
    ImageIndexIVFPQ,
)


def _random_map(num: int, dim: int, seed: int = 0) -> ImageEncodingMap:
    """Build an :class:`ImageEncodingMap` with random, in-memory encodings.

    :param num: Number of gallery vectors.
    :param dim: Dimensionality of each vector.
    :param seed: Seed for reproducible vectors.
    :returns: A populated ``ImageEncodingMap`` not backed by any image files.
    """
    rng = np.random.default_rng(seed)
    return ImageEncodingMap(
        {f"img_{i}.png": rng.random(dim).astype(np.float32) for i in range(num)}
    )


# Construction and exposed state


def test_is_subclass_of_image_index() -> None:
    """Both concrete indexes derive from the :class:`ImageIndex` interface."""
    assert issubclass(ImageIndexIVFFlat, ImageIndex)
    assert issubclass(ImageIndexIVFPQ, ImageIndex)


def test_len_dim_and_paths_match_insertion_order() -> None:
    """The index reports the gallery size, dimensionality and ordered paths."""
    store = _random_map(num=40, dim=8)
    index = ImageIndexIVFFlat(store, nlist=4, nprobe=2)
    assert len(index) == 40
    assert index.dim == 8
    assert index.paths == list(store.keys())


def test_encoding_map_is_exposed() -> None:
    """The source encoding map is reachable from the index."""
    store = _random_map(num=20, dim=8)
    index = ImageIndexIVFFlat(store, nlist=2)
    assert index.encoding_map is store


# Search


def test_search_returns_expected_shapes() -> None:
    """``search`` returns ``(M, k)`` score and id arrays."""
    store = _random_map(num=40, dim=8)
    index = ImageIndexIVFFlat(store, nlist=4, nprobe=4)
    queries = np.asarray(list(store.values()))[:3]
    scores, ids = index.search(queries, k=5)
    assert scores.shape == (3, 5)
    assert ids.shape == (3, 5)


def test_search_accepts_single_vector() -> None:
    """A 1-D query vector is treated as a single-row batch."""
    store = _random_map(num=40, dim=8)
    index = ImageIndexIVFFlat(store, nlist=4, nprobe=4)
    scores, ids = index.search(np.asarray(list(store.values()))[0], k=3)
    assert scores.shape == (1, 3)
    assert ids.shape == (1, 3)


@pytest.mark.parametrize("quantizer", ["l2", "inner_product"])
def test_exhaustive_search_recovers_self(quantizer: str) -> None:
    """With ``nprobe == nlist`` each gallery vector is its own top match."""
    store = _random_map(num=40, dim=8)
    index = ImageIndexIVFFlat(store, quantizer=quantizer, nlist=4, nprobe=4)
    _, ids = index.search(np.asarray(list(store.values())), k=1)
    assert ids[:, 0].tolist() == list(range(40))


def test_search_ids_index_into_paths() -> None:
    """Every returned id is a valid position into :attr:`paths`."""
    store = _random_map(num=40, dim=8)
    index = ImageIndexIVFFlat(store, nlist=4, nprobe=2)
    _, ids = index.search(np.asarray(list(store.values()))[:5], k=3)
    valid = ids[ids >= 0]
    assert valid.min() >= 0
    assert valid.max() < len(index)


def test_search_rejects_non_positive_k() -> None:
    """A non-positive ``k`` is rejected."""
    index = ImageIndexIVFFlat(_random_map(num=20, dim=8), nlist=2)
    with pytest.raises(ValueError, match="'k' must be a positive integer"):
        index.search(np.zeros((1, 8), dtype=np.float32), k=0)


# Metric / normalization behaviour


def test_inner_product_normalizes_gallery_vectors() -> None:
    """Inner-product indexes L2-normalise the gallery before indexing."""
    store = _random_map(num=40, dim=8)
    index = ImageIndexIVFFlat(store, quantizer="inner_product", nlist=4)
    norms = np.linalg.norm(index._vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_l2_leaves_gallery_vectors_unnormalized() -> None:
    """L2 indexes keep the raw gallery vectors."""
    store = _random_map(num=40, dim=8)
    index = ImageIndexIVFFlat(store, quantizer="l2", nlist=4)
    norms = np.linalg.norm(index._vectors, axis=1)
    assert not np.allclose(norms, 1.0)


# Cluster centers


def test_cluster_centers_shape() -> None:
    """``cluster_centers`` returns one centroid per cell."""
    store = _random_map(num=40, dim=8)
    index = ImageIndexIVFFlat(store, nlist=4)
    assert index.cluster_centers.shape == (4, 8)


# Validation


def test_unknown_quantizer_raises() -> None:
    """An unsupported quantizer is rejected."""
    with pytest.raises(ValueError, match="Unsupported quantizer"):
        ImageIndexIVFFlat(_random_map(num=20, dim=8), quantizer="bogus")  # type: ignore[arg-type]


def test_empty_encoding_map_raises() -> None:
    """An empty gallery cannot be indexed."""
    with pytest.raises(ValueError, match="empty ImageEncodingMap"):
        ImageIndexIVFFlat(ImageEncodingMap(), nlist=1)


def test_nlist_larger_than_gallery_raises() -> None:
    """``nlist`` may not exceed the number of indexed vectors."""
    with pytest.raises(ValueError, match="'nlist' must be between 1"):
        ImageIndexIVFFlat(_random_map(num=20, dim=8), nlist=999)


def test_nprobe_out_of_range_raises() -> None:
    """``nprobe`` must lie within ``[1, nlist]``."""
    with pytest.raises(ValueError, match="'nprobe' must be between 1"):
        ImageIndexIVFFlat(_random_map(num=20, dim=8), nlist=4, nprobe=99)


# IVF-PQ specifics


def test_ivfpq_search_and_cluster_centers() -> None:
    """IVF-PQ builds, searches and exposes its coarse centroids."""
    store = _random_map(num=64, dim=16)
    index = ImageIndexIVFPQ(store, nlist=4, nprobe=4, m=4, nbits=4)
    scores, ids = index.search(np.asarray(list(store.values()))[:3], k=5)
    assert scores.shape == (3, 5)
    assert ids.shape == (3, 5)
    assert index.cluster_centers.shape == (4, 16)
    assert (index.m, index.nbits) == (4, 4)


def test_ivfpq_m_must_divide_dim_raises() -> None:
    """``m`` must divide the feature dimensionality."""
    with pytest.raises(ValueError, match="must divide the vector dimensionality"):
        ImageIndexIVFPQ(_random_map(num=64, dim=16), nlist=4, m=5, nbits=4)


def test_ivfpq_too_few_vectors_raises() -> None:
    """The gallery must hold at least ``2 ** nbits`` vectors to train PQ."""
    with pytest.raises(ValueError, match="needs at least"):
        ImageIndexIVFPQ(_random_map(num=20, dim=16), nlist=4, m=4, nbits=8)
