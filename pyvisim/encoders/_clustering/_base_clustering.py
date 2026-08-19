"""
Base classes for the clustering and decomposition models used by the
image embedders.
"""

import abc
from typing import Any, TypeVar

import numpy as np

from ..._errors import NotFittedError
from ...typing import FloatNumpyArray

_ClusteringModelT = TypeVar("_ClusteringModelT", bound="ClusteringModelBase")


def _embed(value: Any) -> Any:
    """
    Recursively converts a fitted-attribute value into JSON-safe data.

    Handles NumPy arrays, NumPy scalars, plain containers and primitives.

    :param value: A value taken from a model's ``__dict__``.
    :return: A JSON-serialisable representation of ``value``.
    :raises TypeError: If ``value`` is of a type that cannot be embedded.
    """
    if isinstance(value, np.ndarray):
        # Preserve the memory order: scikit-learn stores some fitted attributes
        # (e.g. ``PCA.components_``) Fortran-contiguous, and the matrix-product
        # code path differs by layout, so a C-order rebuild would not reproduce
        # the exact same floating-point results.
        order = (
            "F"
            if value.flags["F_CONTIGUOUS"] and not value.flags["C_CONTIGUOUS"]
            else "C"
        )
        return {
            "__ndarray__": True,
            "data": value.tolist(),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "order": order,
        }
    if isinstance(value, np.generic):  # np.float64, np.int64, ...
        return value.item()
    if isinstance(value, dict):
        return {key: _embed(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_embed(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot serialise attribute of type {type(value)!r}.")


def _decode(value: Any) -> Any:
    """
    Rebuilds the fitted-attribute values from JSON data.

    :param value: A value produced by :func:`_embed`, after JSON round-trip.
    :return: The reconstructed value, with arrays restored to ``numpy.ndarray``.
    """
    if isinstance(value, dict):
        if value.get("__ndarray__"):
            array = np.asarray(value["data"], dtype=value["dtype"]).reshape(
                value["shape"]
            )
            if value.get("order") == "F":
                array = np.asfortranarray(array)
            return array
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


class ClusteringModelBase(abc.ABC):
    """Base class for clustering models."""

    @property
    @abc.abstractmethod
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_clusters(self) -> int:
        """Number of clusters (or mixture components) of the model."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_features_in(self) -> int:
        """Number of features the fitted model expects as input."""
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, features: FloatNumpyArray) -> None:
        """
        Fits the model on the given feature matrix.

        :param features: Feature matrix of shape (n_samples, n_features).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Serialises the fitted model into a JSON-safe dictionary.

        :return: A dictionary describing the model class and its fitted
            state required for inference.
        :raises NotFittedError: If the model is not fitted.
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def from_dict(
        cls: type[_ClusteringModelT], data: dict[str, Any]
    ) -> _ClusteringModelT:
        """
        Rebuilds a model from a dictionary produced by :meth:`to_dict`.

        :param data: A mapping with the form: {"__class__": str, "__module__": str, "state": dict}.
        :return: A fitted model.
        """
        raise NotImplementedError

    def _check_is_fitted(self) -> None:
        """
        Ensures the model is fitted before accessing fitted-only attributes.

        :raises NotFittedError: If the model is not fitted.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate data before using this attribute."
            )
