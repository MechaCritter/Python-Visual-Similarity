"""Tests for the clustering base classes, exercised via concrete models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyvisim._errors import NotFittedError
from pyvisim.classic._clustering import (
    PCA,
    ClusteringModelBase,
    DiagCovarGaussianMixture,
    FittedModelBase,
    KMeans,
)


def _fitted(
    model_cls: type[KMeans | DiagCovarGaussianMixture | PCA],
) -> KMeans | DiagCovarGaussianMixture | PCA:
    """A two-cluster (or two-component) model fitted on random features."""
    model = model_cls(2)  # type: ignore[call-arg]
    model.fit(np.random.default_rng(0).normal(size=(40, 6)).astype(np.float32))
    return model


@pytest.mark.parametrize("base_cls", [FittedModelBase, ClusteringModelBase])
def test_model_bases_are_abstract(base_cls: type[FittedModelBase]) -> None:
    """The model base classes cannot be instantiated directly."""
    with pytest.raises(TypeError):
        base_cls()  # type: ignore[abstract]


def test_check_is_fitted_message() -> None:
    """The shared fitted check raises a message mentioning the unfitted state."""
    with pytest.raises(NotFittedError, match="is not fitted yet"):
        _ = KMeans(4).cluster_centers


def test_not_fitted_error_is_value_and_attribute_error() -> None:
    """``NotFittedError`` keeps the sklearn exception hierarchy it replaced."""
    assert issubclass(NotFittedError, ValueError)
    assert issubclass(NotFittedError, AttributeError)


def test_a_serializable_model_must_declare_its_format_version() -> None:
    """A model implementing the state contract has to say which layout it writes."""
    with pytest.raises(TypeError, match="__format_version__"):

        class Unversioned(ClusteringModelBase):
            def _state(self) -> dict[str, Any]:
                return {}

            @classmethod
            def from_dict(cls, state: dict[str, Any], **kwargs: Any) -> Unversioned:
                return cls()


@pytest.mark.parametrize("model_cls", [KMeans, DiagCovarGaussianMixture, PCA])
def test_every_model_writes_its_format_version(
    model_cls: type[KMeans | DiagCovarGaussianMixture | PCA],
) -> None:
    """The serialized dictionary carries the version of the class that wrote it."""
    assert (
        _fitted(model_cls).to_dict()["format_version"] == model_cls.__format_version__
    )


@pytest.mark.parametrize("model_cls", [KMeans, DiagCovarGaussianMixture, PCA])
def test_every_model_keeps_its_dictionary_layout(
    model_cls: type[KMeans | DiagCovarGaussianMixture | PCA],
) -> None:
    """The keys the embedders nest in their files stay the same."""
    assert set(_fitted(model_cls).to_dict()) == {
        "format_version",
        "__class__",
        "__module__",
        "state",
    }


@pytest.mark.parametrize("model_cls", [KMeans, DiagCovarGaussianMixture, PCA])
def test_every_model_round_trips_through_a_file(
    model_cls: type[KMeans | DiagCovarGaussianMixture | PCA], tmp_path: Path
) -> None:
    """A model saved on its own loads back with identical fitted arrays."""
    model = _fitted(model_cls)
    path = model.save_to_disk(tmp_path / "model")
    assert path == tmp_path / "model"

    reloaded = model_cls.load_from_disk(path)

    original_state = model.to_dict()["state"]
    for key, value in reloaded.to_dict()["state"].items():
        assert value == original_state[key]


def test_a_model_file_of_another_class_is_rejected(tmp_path: Path) -> None:
    """Loading a KMeans file as a PCA names the class that wrote it."""
    path = _fitted(KMeans).save_to_disk(tmp_path / "model")
    with pytest.raises(ValueError, match="KMeans"):
        PCA.load_from_disk(path)


def test_from_dict_rejects_keyword_arguments() -> None:
    """A model rebuilds from its dictionary alone."""
    with pytest.raises(TypeError, match="'rng'"):
        KMeans.from_dict(_fitted(KMeans).to_dict(), rng=0)


def test_a_dictionary_without_a_format_version_still_loads() -> None:
    """Dictionaries written before the models carried a version stay readable."""
    model = KMeans(2)
    model.fit(np.random.default_rng(0).normal(size=(40, 6)).astype(np.float32))
    data = model.to_dict()
    del data["format_version"]
    np.testing.assert_array_equal(
        KMeans.from_dict(data).cluster_centers, model.cluster_centers
    )
