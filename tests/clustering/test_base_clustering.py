"""Tests for the clustering base classes, exercised via concrete models."""

from __future__ import annotations

import pytest
from sklearn.exceptions import NotFittedError

from pyvisim.clustering import PCA, ClusteringModelBase, KMeans


def test_clustering_base_is_abstract() -> None:
    """``ClusteringModelBase`` cannot be instantiated directly (it is abstract)."""
    with pytest.raises(TypeError):
        ClusteringModelBase()  # type: ignore[abstract]


def test_repr_contains_model() -> None:
    """The shared ``__repr__`` names the concrete class and its model."""
    text = repr(KMeans(8))
    assert "KMeans(" in text
    assert "model=" in text


def test_check_is_fitted_message() -> None:
    """The shared fitted check raises a message mentioning the unfitted state."""
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        _ = PCA(4).n_components
