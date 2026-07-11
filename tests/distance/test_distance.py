"""Tests for the pure-NumPy pairwise metrics in :mod:`pyvisim.distance`."""

from __future__ import annotations

import os

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sklearn.metrics.pairwise import euclidean_distances as sk_euclidean
from sklearn.metrics.pairwise import manhattan_distances as sk_manhattan

from pyvisim.distance import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)

#: Shapes (N, M, D) exercised against the scikit-learn reference.
PAIRWISE_SHAPES = [(1, 1, 2), (3, 5, 8), (10, 7, 64), (4, 4, 128)]

#: Gallery size of the large-input stress tests (marked ``slow``). Override
#: via environment variables to fit the machine, e.g.
#: ``PYVISIM_TEST_LARGE_ROWS=20000 PYVISIM_TEST_LARGE_FEATURES=1000``.
LARGE_ROWS = int(os.environ.get("PYVISIM_TEST_LARGE_ROWS", "100000"))
LARGE_FEATURES = int(os.environ.get("PYVISIM_TEST_LARGE_FEATURES", "10000"))
#: Query rows in the large tests, kept small so the (N, M) output stays cheap
#: while N x D is pushed to the configured size.
LARGE_QUERY_ROWS = 8


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator so failures are reproducible."""
    return np.random.default_rng(42)


@pytest.mark.parametrize(("n", "m", "d"), PAIRWISE_SHAPES)
def test_cosine_matches_sklearn(
    rng: np.random.Generator, n: int, m: int, d: int
) -> None:
    x = rng.standard_normal((n, d))
    y = rng.standard_normal((m, d))
    np.testing.assert_allclose(cosine_similarity(x, y), sk_cosine(x, y), atol=1e-12)


@pytest.mark.parametrize(("n", "m", "d"), PAIRWISE_SHAPES)
def test_euclidean_matches_sklearn(
    rng: np.random.Generator, n: int, m: int, d: int
) -> None:
    x = rng.standard_normal((n, d))
    y = rng.standard_normal((m, d))
    np.testing.assert_allclose(euclidean_distances(x, y), sk_euclidean(x, y), atol=1e-9)


@pytest.mark.parametrize(("n", "m", "d"), PAIRWISE_SHAPES)
def test_manhattan_matches_sklearn(
    rng: np.random.Generator, n: int, m: int, d: int
) -> None:
    x = rng.standard_normal((n, d))
    y = rng.standard_normal((m, d))
    np.testing.assert_allclose(
        manhattan_distances(x, y), sk_manhattan(x, y), atol=1e-12
    )


def test_all_metrics_return_float64(rng: np.random.Generator) -> None:
    x = rng.standard_normal((3, 4)).astype(np.float32)
    y = rng.standard_normal((2, 4)).astype(np.float32)
    for metric in (cosine_similarity, euclidean_distances, manhattan_distances):
        result = metric(x, y)
        assert result.dtype == np.float64
        assert result.shape == (3, 2)


def test_cosine_zero_vector_yields_zero_not_nan() -> None:
    x = np.zeros((1, 4))
    y = np.ones((2, 4))
    result = cosine_similarity(x, y)
    np.testing.assert_array_equal(result, np.zeros((1, 2)))


def test_cosine_identical_rows_are_exactly_similar(
    rng: np.random.Generator,
) -> None:
    x = rng.standard_normal((5, 16))
    np.testing.assert_allclose(np.diag(cosine_similarity(x, x)), 1.0, atol=1e-12)


def test_euclidean_self_distance_diagonal_is_exactly_zero(
    rng: np.random.Generator,
) -> None:
    # Scale up so that naive float arithmetic would leave rounding residue on
    # the diagonal; the same-object fast path must still return exact zeros.
    x = rng.standard_normal((6, 32)) * 1e6
    assert (np.diag(euclidean_distances(x, x)) == 0.0).all()


def test_euclidean_is_never_negative(rng: np.random.Generator) -> None:
    x = rng.standard_normal((4, 8)) + 1e8
    assert (euclidean_distances(x, x + 1e-8) >= 0.0).all()


def test_euclidean_float32_input_does_not_lose_precision() -> None:
    # Large offsets make the dot-product expansion cancel catastrophically in
    # float32; the float64 upcast has to preserve the small true distance.
    x = np.array([[10000.0, 10000.0]], dtype=np.float32)
    y = np.array([[10001.0, 10001.0]], dtype=np.float32)
    np.testing.assert_allclose(euclidean_distances(x, y), [[np.sqrt(2.0)]], rtol=1e-6)


def test_manhattan_chunked_path_matches_unchunked(
    rng: np.random.Generator,
) -> None:
    # A working-memory budget of one byte forces every row into its own
    # chunk; the result must be identical to the single-chunk path.
    x = rng.standard_normal((9, 12))
    y = rng.standard_normal((5, 12))
    expected = manhattan_distances(x, y)
    np.testing.assert_array_equal(
        manhattan_distances(x, y, working_memory_bytes=1), expected
    )


def test_manhattan_rejects_non_positive_working_memory(
    rng: np.random.Generator,
) -> None:
    x = rng.standard_normal((2, 3))
    with pytest.raises(ValueError, match="working_memory_bytes"):
        manhattan_distances(x, x, working_memory_bytes=0)


@pytest.mark.parametrize(
    "metric", [cosine_similarity, euclidean_distances, manhattan_distances]
)
def test_non_2d_input_raises(metric: object) -> None:
    with pytest.raises(ValueError, match="2-D"):
        metric(np.ones(4), np.ones((2, 4)))  # type: ignore[operator]


@pytest.mark.parametrize(
    "metric", [cosine_similarity, euclidean_distances, manhattan_distances]
)
def test_mismatched_feature_dims_raise(metric: object) -> None:
    with pytest.raises(ValueError, match="feature"):
        metric(np.ones((2, 3)), np.ones((2, 4)))  # type: ignore[operator]


@pytest.fixture(scope="module")
def large_inputs() -> tuple[np.ndarray, np.ndarray]:
    """A ``(LARGE_ROWS, LARGE_FEATURES)`` gallery and a small query matrix.

    Module-scoped so the multi-gigabyte gallery is generated once and shared
    by all large-input tests.
    """
    rng = np.random.default_rng(7)
    x = rng.standard_normal((LARGE_ROWS, LARGE_FEATURES))
    y = rng.standard_normal((LARGE_QUERY_ROWS, LARGE_FEATURES))
    return x, y


def _spot_rows(n_rows: int) -> list[int]:
    """A handful of row indices spread across a large matrix."""
    return sorted({0, n_rows // 3, n_rows // 2, n_rows - 1})


@pytest.mark.slow
def test_cosine_large_inputs(large_inputs: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = large_inputs
    result = cosine_similarity(x, y)
    assert result.shape == (LARGE_ROWS, LARGE_QUERY_ROWS)
    # Spot-check rows against a direct per-row computation; comparing the
    # full matrix to scikit-learn would double the multi-GB footprint.
    for i in _spot_rows(LARGE_ROWS):
        expected = (y @ x[i]) / (np.linalg.norm(x[i]) * np.linalg.norm(y, axis=1))
        np.testing.assert_allclose(result[i], expected, rtol=1e-10)


@pytest.mark.slow
def test_euclidean_large_inputs(large_inputs: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = large_inputs
    result = euclidean_distances(x, y)
    assert result.shape == (LARGE_ROWS, LARGE_QUERY_ROWS)
    for i in _spot_rows(LARGE_ROWS):
        expected = np.linalg.norm(x[i] - y, axis=1)
        np.testing.assert_allclose(result[i], expected, rtol=1e-9)


@pytest.mark.slow
def test_manhattan_large_inputs(large_inputs: tuple[np.ndarray, np.ndarray]) -> None:
    x, y = large_inputs
    # The default 256 MiB budget forces the large gallery through many
    # chunks, so this exercises the chunked path end to end.
    result = manhattan_distances(x, y)
    assert result.shape == (LARGE_ROWS, LARGE_QUERY_ROWS)
    for i in _spot_rows(LARGE_ROWS):
        expected = np.abs(x[i] - y).sum(axis=1)
        np.testing.assert_allclose(result[i], expected, rtol=1e-12)
