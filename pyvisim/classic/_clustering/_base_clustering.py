"""
Base classes for the clustering and decomposition models used by the
image embedders.
"""

import abc
from typing import Any, ClassVar

import numpy as np

from ..._errors import NotFittedError
from ...serialization import SerializerMixin, decode_array_node
from ...typing import FloatNumpyArray


def _embed(value: Any) -> Any:
    """
    Recursively converts a fitted-attribute value into JSON-safe data.

    Handles NumPy arrays, NumPy scalars, plain containers and primitives.

    :param value: A value taken from a model's ``__dict__``.
    :return: A JSON-serializable representation of ``value``.
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
    raise TypeError(f"Cannot serialize attribute of type {type(value)!r}.")


def _decode(value: Any) -> Any:
    """
    Rebuilds the fitted-attribute values from JSON data.

    :param value: A value produced by :func:`_embed`, after JSON round-trip.
    :return: The reconstructed value, with arrays restored to ``numpy.ndarray``.
    """
    if isinstance(value, dict):
        if value.get("__ndarray__"):
            return decode_array_node(value)
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


class FittedModelBase(SerializerMixin):
    """
    Base class for the models the image embedders fit on local features.

    A model serializes into ``{"format_version": int, "__class__": str,
    "__module__": str, "state": dict}``. ``format_version`` is absent from
    dictionaries written before the models carried a version.
    """

    __file_format__: ClassVar[str] = ".safetensors"
    __metadata_key__: ClassVar[str] = "pyvisim_model"
    __state_keys__: ClassVar[frozenset[str]] = frozenset({"state"})

    @property
    @abc.abstractmethod
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
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

    @staticmethod
    def _validate_features(
        features: FloatNumpyArray, *, n_features: int | None = None
    ) -> FloatNumpyArray:
        """
        Checks that the given features form a non-empty 2-D feature matrix.

        :param features: Feature matrix of shape (n_samples, n_features).
        :param n_features: Number of features every sample must have, or
            ``None`` to accept any.
        :return: The features as a NumPy array, in their own dtype.
        :raises ValueError: If the matrix is not 2-D, holds no sample, or its
            feature count differs from ``n_features``.
        """
        data = np.asarray(features)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got a {data.ndim}D array.")
        if data.shape[0] == 0:
            raise ValueError("Expected a non-empty feature matrix, got 0 samples.")
        if n_features is not None and data.shape[1] != n_features:
            raise ValueError(
                f"Expected {n_features} features per sample, got {data.shape[1]}."
            )
        return data

    def _wrap_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Puts the fitted attributes of this model under the ``"state"`` key.

        :param state: The fitted attributes, arrays included.
        :return: A JSON-safe mapping holding the module name and ``state``.
        """
        return {"__module__": type(self).__module__, "state": _embed(state)}

    @classmethod
    def _unwrap_state(cls, data: dict[str, Any], *legacy_names: str) -> dict[str, Any]:
        """
        Validates a serialized model and decodes its fitted attributes.

        :param data: A mapping produced by :meth:`to_dict`.
        :param legacy_names: Class names older releases wrote for this model.
        :return: The fitted attributes, with arrays restored to
            ``numpy.ndarray``.
        :raises TypeError: If ``data`` is not a dictionary.
        :raises ValueError: If ``data`` is malformed or describes a different
            model type than this class expects.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected a dict from to_dict(), got {type(data).__name__}."
            )
        for key in ("__class__", "state"):
            if key not in data:
                raise ValueError(f"Malformed model dict; missing key {key!r}.")
        if data["__class__"] not in (cls.__name__, *legacy_names):
            raise ValueError(
                f"{cls.__name__} expects a serialized {cls.__name__!r}, "
                f"got {data['__class__']!r}."
            )
        state: dict[str, Any] = _decode(data["state"])
        return state


class ClusteringModelBase(FittedModelBase):
    """Base class for clustering models."""

    @property
    @abc.abstractmethod
    def n_clusters(self) -> int:
        """Number of clusters (or mixture components) of the model."""
        raise NotImplementedError
