"""
Base classes for the scikit-learn-backed models used by the image encoders.
"""

import abc
from typing import Any, ClassVar, TypeVar

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from ..typing import FloatNumpyArray

_SklearnModelT = TypeVar("_SklearnModelT", bound="_SklearnModelBase")


def _encode(value: Any) -> Any:
    """
    Recursively converts a fitted-attribute value into JSON-safe data.

    Handles the types scikit-learn stores on its estimators: NumPy arrays,
    NumPy scalars, plain containers and primitives.

    :param value: A value taken from an estimator's ``__dict__``.
    :return: A JSON-serialisable representation of ``value``.
    :raises TypeError: If ``value`` is of a type that cannot be encoded.
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
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot serialise attribute of type {type(value)!r}.")


def _decode(value: Any) -> Any:
    """
    Rebuilds the scikit-learn objects from JSON data.

    :param value: A value produced by :func:`_encode`, after JSON round-trip.
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


class _SklearnModelBase(abc.ABC):
    """
    Base class for models backed by a scikit-learn estimator.

    :param model: Underlying scikit-learn estimator instance.
    """

    _sklearn_class: ClassVar[type[Any]]

    #: Fitted attributes that are training artifacts, not needed for inference.
    #: They are excluded from :meth:`to_dict` to keep serialised models small.
    _transient_attrs: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, model: Any) -> None:
        self._model = model

    @property
    def is_fitted(self) -> bool:
        """Whether the underlying estimator has been fitted."""
        try:
            check_is_fitted(self._model)
        except NotFittedError:
            return False
        return True

    def _check_is_fitted(self) -> None:
        """
        Ensures the underlying estimator is fitted before accessing
        fitted-only attributes.

        :raises NotFittedError: If the underlying estimator is not fitted.
        """
        if not self.is_fitted:
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate data before using this attribute."
            )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialises the fitted estimator into a JSON-safe dictionary.

        The returned mapping contains only plain numbers, strings and nested
        containers.

        **NOTE**: Training-only artifacts listed in :attr:`_transient_attrs`
        (e.g. K-Means ``labels_``) are excluded, since they are not needed to
        use the fitted estimator and would bloat the serialised file.

        :return: A dictionary describing the underlying estimator class and
            its fitted state required for inference.
        :raises NotFittedError: If the underlying estimator is not fitted.
        """
        self._check_is_fitted()
        state = {
            key: value
            for key, value in vars(self._model).items()
            if key not in self._transient_attrs
        }
        return {
            "__class__": type(self._model).__name__,
            "__module__": type(self._model).__module__,
            "state": _encode(state),
        }

    @classmethod
    def from_dict(cls: type[_SklearnModelT], data: dict[str, Any]) -> _SklearnModelT:
        """
        Rebuilds a model from a dictionary produced by :meth:`to_dict`.

        :param data: A mapping with the form: {"__class__": str, "__module__": str, "state": dict}.
        :return: A fitted model backed by the reconstructed estimator.
        :raises TypeError: If ``data`` is not a dictionary.
        :raises ValueError: If ``data`` is malformed or describes a different
            estimator type than this class expects.
        """
        if not isinstance(data, dict):
            raise TypeError(
                f"Expected a dict from to_dict(), got {type(data).__name__}."
            )

        for key in ("__class__", "state"):
            if key not in data:
                raise ValueError(f"Malformed model dict; missing key {key!r}.")

        expected = cls._sklearn_class.__name__
        if data["__class__"] != expected:
            raise ValueError(
                f"{cls.__name__} expects a serialised {expected!r}, "
                f"got {data['__class__']!r}."
            )

        # Build the estimator without calling ``__init__`` and restore its
        # fitted state directly; ``_sklearn_class`` is dynamic, hence ``Any``.
        sklearn_class: Any = cls._sklearn_class
        model = sklearn_class.__new__(sklearn_class)
        model.__dict__.update(_decode(data["state"]))
        return cls._from_sklearn(model)

    @property
    def n_features_in(self) -> int:
        """
        Number of features the fitted estimator expects as input.

        :raises NotFittedError: If the underlying estimator is not fitted.
        """
        self._check_is_fitted()
        return int(self._model.n_features_in_)

    def fit(self, features: FloatNumpyArray) -> None:
        """
        Fits the underlying estimator on the given feature matrix.

        :param features: Feature matrix of shape (n_samples, n_features).
        """
        self._model.fit(features)

    def _validate_sklearn_model(self) -> None:  # noqa: B027
        """
        Hook for subclasses to validate (or coerce) an estimator that was
        passed in directly via :meth:`_from_sklearn`.
        """

    @classmethod
    def _from_sklearn(cls: type[_SklearnModelT], model: Any) -> _SklearnModelT:
        """
        Creates a model from an existing scikit-learn estimator.

        This is an internal constructor used to adopt pretrained estimators
        (loaded from legacy weight files).

        :param model: Estimator instance of the underlying scikit-learn class.
        :return: A model backed by the given estimator.
        :raises TypeError: If ``model`` is not an instance of the underlying class.
        """
        if not isinstance(model, cls._sklearn_class):
            raise TypeError(
                f"{cls.__name__} can only be created from instances of "
                f"{cls._sklearn_class.__name__}, not {type(model).__name__}."
            )
        instance = cls.__new__(cls)
        _SklearnModelBase.__init__(instance, model)
        instance._validate_sklearn_model()
        return instance

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model!r})"


class ClusteringModelBase(_SklearnModelBase):
    """
    Base class for clustering models.
    """

    @property
    @abc.abstractmethod
    def n_clusters(self) -> int:
        """Number of clusters (or mixture components) of the estimator."""
        raise NotImplementedError
