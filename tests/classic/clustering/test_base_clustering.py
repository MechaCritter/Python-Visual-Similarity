"""Tests for the clustering base classes, exercised via concrete models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyvisim._errors import NotFittedError
from pyvisim.classic._clustering import (
    PCA,
    ClusteringModelBase,
    DiagCovarGaussianMixture,
    KMeans,
)


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


def test_a_serialisable_model_must_declare_its_format_version() -> None:
    """A model implementing ``to_dict`` has to say which layout it writes."""
    with pytest.raises(TypeError, match="__format_version__"):

        class Unversioned(ClusteringModelBase):
            def to_dict(self) -> dict[str, Any]:
                return {}


@pytest.mark.parametrize("model_cls", [KMeans, DiagCovarGaussianMixture, PCA])
def test_every_model_writes_its_format_version(
    model_cls: type[KMeans | DiagCovarGaussianMixture | PCA],
) -> None:
    """The serialised dictionary carries the version of the class that wrote it."""
    model = model_cls(2)  # type: ignore[call-arg]
    model.fit(np.random.default_rng(0).normal(size=(40, 6)).astype(np.float32))
    assert model.to_dict()["format_version"] == model_cls.__format_version__


def test_a_dictionary_without_a_format_version_still_loads() -> None:
    """Dictionaries written before the models carried a version stay readable."""
    model = KMeans(2)
    model.fit(np.random.default_rng(0).normal(size=(40, 6)).astype(np.float32))
    data = model.to_dict()
    del data["format_version"]
    np.testing.assert_array_equal(
        KMeans.from_dict(data).cluster_centers, model.cluster_centers
    )
