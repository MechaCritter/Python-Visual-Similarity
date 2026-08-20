"""Tests for the clustering base classes, exercised via concrete models."""

from __future__ import annotations

import pytest

from pyvisim._errors import NotFittedError
from pyvisim.classic._clustering import ClusteringModelBase, KMeans


def test_clustering_base_is_abstract() -> None:
    """``ClusteringModelBase`` cannot be instantiated directly (it is abstract)."""
    with pytest.raises(TypeError):
        ClusteringModelBase()  # type: ignore[abstract]


def test_check_is_fitted_message() -> None:
    """The shared fitted check raises a message mentioning the unfitted state."""
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        _ = KMeans(4).cluster_centers


def test_not_fitted_error_is_value_and_attribute_error() -> None:
    """``NotFittedError`` keeps the sklearn exception hierarchy it replaced."""
    assert issubclass(NotFittedError, ValueError)
    assert issubclass(NotFittedError, AttributeError)
