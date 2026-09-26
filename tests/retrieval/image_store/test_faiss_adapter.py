"""Tests for the adapter the image store searches a FAISS index through."""

from __future__ import annotations

import sys
import tracemalloc
from typing import Any

import numpy as np
import pytest

from pyvisim.retrieval.image_store import BruteForceIndex
from pyvisim.retrieval.image_store._index._adapter import (
    FaissIndexAdapter,
    is_faiss_index,
)

faiss = pytest.importorskip("faiss")


@pytest.fixture(scope="module")
def vectors() -> np.ndarray:
    """A ``(30, 8)`` gallery matrix, L2-normalised for inner-product search.

    :returns: a float32 matrix ready to be added to a FAISS index.
    """
    rng = np.random.default_rng(0)
    matrix = rng.random((30, 8), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.ascontiguousarray(matrix)


@pytest.fixture
def flat_index(vectors: np.ndarray) -> Any:
    """A flat inner-product FAISS index over the gallery.

    :param vectors: the gallery matrix.
    :returns: a populated ``faiss.IndexFlatIP``.
    """
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def _ivf_index(vectors: np.ndarray) -> Any:
    """An IVF-Flat index over the gallery, without a direct map.

    :param vectors: the gallery matrix.
    :returns: a trained and populated ``faiss.IndexIVFFlat``.
    """
    quantizer = faiss.IndexFlatL2(vectors.shape[1])
    ivf = faiss.IndexIVFFlat(quantizer, vectors.shape[1], 4)
    ivf.train(vectors)
    ivf.add(vectors)
    return ivf


# Recognizing a FAISS index


def test_recognizes_a_faiss_index(flat_index: Any, vectors: np.ndarray) -> None:
    """Only a ``faiss.Index`` counts as a FAISS index."""
    assert is_faiss_index(flat_index)
    assert not is_faiss_index(BruteForceIndex(vectors))
    assert not is_faiss_index(vectors)


def test_recognizing_an_index_needs_faiss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without FAISS installed, the check asks for it to be installed."""
    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(ImportError, match="FAISS is not installed"):
        is_faiss_index(object())


# Construction and exposed state


def test_reconstructs_the_vectors_of_a_flat_index(
    flat_index: Any, vectors: np.ndarray
) -> None:
    """A flat index hands its vectors back as they were added."""
    adapter = FaissIndexAdapter(flat_index)
    assert np.allclose(adapter.vectors, vectors, atol=1e-6)
    assert len(adapter) == vectors.shape[0]
    assert adapter.dim == vectors.shape[1]


def test_keeps_the_faiss_index(flat_index: Any) -> None:
    """The FAISS index stays reachable for its own API."""
    assert FaissIndexAdapter(flat_index).faiss_index is flat_index


@pytest.mark.parametrize(
    ("factory", "metric", "name"),
    [
        ("Flat", faiss.METRIC_INNER_PRODUCT, "faiss.IndexFlat"),
        ("HNSW8,Flat", faiss.METRIC_L2, "faiss.IndexHNSWFlat"),
        ("PQ4x4", faiss.METRIC_L2, "faiss.IndexPQ"),
    ],
)
def test_is_named_after_the_index_type(
    vectors: np.ndarray, factory: str, metric: int, name: str
) -> None:
    """The name is the class of the FAISS index, whatever built it."""
    index = faiss.index_factory(vectors.shape[1], factory, metric)
    index.train(vectors)
    index.add(vectors)
    adapter = FaissIndexAdapter(index)
    assert adapter.name == name
    assert repr(adapter).startswith(f"FaissIndexAdapter(name={name!r}")


@pytest.mark.parametrize(
    ("metric", "space"),
    [(faiss.METRIC_L2, "l2"), (faiss.METRIC_INNER_PRODUCT, "ip")],
)
def test_space_follows_the_metric(vectors: np.ndarray, metric: int, space: str) -> None:
    """The metric of the index decides the space its scores are distances of."""
    index = faiss.IndexFlat(vectors.shape[1], metric)
    index.add(vectors)
    assert FaissIndexAdapter(index).space == space


def test_rejects_an_unsupported_metric(vectors: np.ndarray) -> None:
    """A metric without a matching space is rejected."""
    index = faiss.IndexFlat(vectors.shape[1], faiss.METRIC_L1)
    index.add(vectors)
    with pytest.raises(ValueError, match="METRIC_L2 and METRIC_INNER_PRODUCT"):
        FaissIndexAdapter(index)


def test_rejects_an_empty_index() -> None:
    """An index without vectors has no gallery to search."""
    with pytest.raises(ValueError, match="holds no vectors"):
        FaissIndexAdapter(faiss.IndexFlatIP(8))


# Reading the vectors back


def test_vectors_are_read_only(flat_index: Any) -> None:
    """The gallery the adapter hands out cannot be written to."""
    adapter = FaissIndexAdapter(flat_index)
    with pytest.raises(ValueError, match="read-only"):
        adapter.vectors[0, 0] = 1.0


def test_vectors_at_reads_the_requested_rows(
    flat_index: Any, vectors: np.ndarray
) -> None:
    """``vectors_at`` hands back the named rows of the gallery, read-only."""
    adapter = FaissIndexAdapter(flat_index)
    block = adapter.vectors_at([4, 1])
    assert np.allclose(block, vectors[[4, 1]], atol=1e-6)
    assert not block.flags.writeable
    for bad_ids in ([], [vectors.shape[0]]):
        with pytest.raises(ValueError, match="'ids'"):
            adapter.vectors_at(bad_ids)


def test_reads_the_vectors_back_without_keeping_a_copy() -> None:
    """The adapter allocates no gallery of its own."""
    gallery = np.random.default_rng(1).random((4000, 64), dtype=np.float32)
    flat = faiss.IndexFlatIP(gallery.shape[1])
    flat.add(gallery)

    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        adapter = FaissIndexAdapter(flat)
        after, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert after - before < gallery.nbytes // 100
    assert np.array_equal(adapter.vectors, gallery)


def test_ivf_index_without_a_direct_map_is_rejected(vectors: np.ndarray) -> None:
    """An IVF index that cannot look a row up points at its direct map."""
    with pytest.raises(ValueError, match="make_direct_map"):
        FaissIndexAdapter(_ivf_index(vectors))


def test_ivf_index_reads_its_vectors_after_a_direct_map(vectors: np.ndarray) -> None:
    """An IVF-Flat index with a direct map hands its vectors back as added."""
    ivf = _ivf_index(vectors)
    faiss.extract_index_ivf(ivf).make_direct_map()

    adapter = FaissIndexAdapter(ivf)
    assert np.allclose(adapter.vectors, vectors, atol=1e-6)
    assert np.allclose(adapter.vectors_at([7, 2]), vectors[[7, 2]], atol=1e-6)


def test_id_map_index_is_rejected(vectors: np.ndarray) -> None:
    """An ID-mapped index cannot look a row up, so it cannot be adapted."""
    id_map = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
    id_map.add_with_ids(vectors, np.arange(vectors.shape[0]))

    with pytest.raises(ValueError, match="cannot look its vectors up by row"):
        FaissIndexAdapter(id_map)


def test_compressed_index_reconstructs_an_approximation(vectors: np.ndarray) -> None:
    """A product-quantized index gives back a lossy version of its gallery."""
    pq = faiss.IndexPQ(vectors.shape[1], 4, 4)
    pq.train(vectors)
    pq.add(vectors)

    adapter = FaissIndexAdapter(pq)
    assert adapter.vectors.shape == vectors.shape
    assert not np.allclose(adapter.vectors, vectors, atol=1e-6)


# Searching


@pytest.mark.parametrize(
    ("metric", "space"),
    [(faiss.METRIC_L2, "l2"), (faiss.METRIC_INNER_PRODUCT, "ip")],
)
def test_scores_match_the_built_in_index(
    vectors: np.ndarray, metric: int, space: str
) -> None:
    """A FAISS index scores like the built-in index of the same space."""
    index = faiss.IndexFlat(vectors.shape[1], metric)
    index.add(vectors)

    scores, ids = FaissIndexAdapter(index).search(vectors[:5], k=6)
    built_in_scores, built_in_ids = BruteForceIndex(vectors, space=space).search(
        vectors[:5], k=6
    )
    assert np.array_equal(ids, built_in_ids)
    assert np.allclose(scores, built_in_scores, atol=1e-5)


def test_inner_product_scores_are_distances(
    flat_index: Any, vectors: np.ndarray
) -> None:
    """A self-match scores zero and the scores grow along a result row."""
    scores, ids = FaissIndexAdapter(flat_index).search(vectors[:3], k=4)
    assert scores.dtype == np.float32
    assert np.array_equal(ids[:, 0], np.arange(3))
    assert np.allclose(scores[:, 0], 0.0, atol=1e-5)
    assert (np.diff(scores, axis=1) >= -1e-6).all()


def test_missing_neighbors_are_padded(vectors: np.ndarray) -> None:
    """A gallery smaller than ``k`` pads with the id ``-1`` and an infinite score."""
    small = faiss.IndexFlatIP(vectors.shape[1])
    small.add(vectors[:3])

    scores, ids = FaissIndexAdapter(small).search(vectors[:1], k=5)
    assert list(ids[0, 3:]) == [-1, -1]
    assert np.isinf(scores[0, 3:]).all()
    assert np.isfinite(scores[0, :3]).all()


def test_search_accepts_a_single_vector(flat_index: Any, vectors: np.ndarray) -> None:
    """A ``(D,)`` query is searched as a batch of one."""
    scores, ids = FaissIndexAdapter(flat_index).search(vectors[0], k=2)
    assert scores.shape == (1, 2)
    assert ids[0, 0] == 0


def test_search_rejects_a_wrong_dimensionality(flat_index: Any) -> None:
    """A query whose width differs from the indexed one is rejected."""
    with pytest.raises(ValueError, match="dimensionality"):
        FaissIndexAdapter(flat_index).search(np.zeros((1, 3), dtype=np.float32), k=2)


def test_search_rejects_a_non_positive_k(flat_index: Any, vectors: np.ndarray) -> None:
    """``k`` must be a positive integer."""
    with pytest.raises(ValueError, match="'k' must be >= 1"):
        FaissIndexAdapter(flat_index).search(vectors[:1], k=0)
